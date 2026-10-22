"""Explicit v2 MLP alternatives for the bounded thermal-plate problem.

This module preserves the v1 global soft PINN.  The optional revised field
uses three MLP subdomains and exact top/bottom boundary lifting; its extra
interface temperature and Fourier-flux conditions are independently checked.
No reference solver or hidden temperature labels are imported or trained on.
"""
from dataclasses import dataclass, asdict
import math
import hashlib
from time import perf_counter
import numpy as np
import torch
from torch import nn
from ..physics.planar_contact import PlanarContactConfig
from .planar_contact_inverse import PlanarObservations


@dataclass(frozen=True)
class PlanarRevisionConfig:
    architecture: str = "hard_boundary_decomposed"
    width: int = 32
    layers: int = 3
    adam_epochs: int = 1800
    lbfgs_iterations: int = 1000
    interior_points: int = 768
    boundary_points: int = 128
    interface_points: int = 128
    refresh_adam_every: int = 100
    learning_rate: float = .001
    physics_weight: float = 100.
    observation_weight: float = 1.
    interface_temperature_scale: float = .1
    temporal_envelope: str = "linear"
    time_feature: str = "linear"
    skip_hard_bc_training: bool = False
    duration: float = 120.
    initial_temperature: float = 300.
    temperature_scale: float = 5.
    ramp_seconds: float = 10.
    bath_drop_kelvin: float = 5.
    top_flux_watts_per_square_metre: float = 500.
    h_bounds: tuple = (400., 2000.)
    initial_h: float = math.sqrt(400.*2000.)

    def __post_init__(self):
        if self.architecture not in ("soft_mlp", "hard_boundary_decomposed"):
            raise ValueError("unknown revision architecture")
        if self.temporal_envelope not in ("linear","ramp") or self.time_feature not in ("linear","log_diffusion"):
            raise ValueError("unknown temporal conditioning")
        for key in ("width", "layers", "adam_epochs", "interior_points", "boundary_points", "interface_points", "refresh_adam_every"):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.lbfgs_iterations) is not int or self.lbfgs_iterations<0:
            raise ValueError("LBFGS cap must be nonnegative")
        if not math.isfinite(self.observation_weight) or self.observation_weight<0:
            raise ValueError("observation weight must be finite and nonnegative")
        for key in ("learning_rate","physics_weight","interface_temperature_scale","duration","initial_temperature","temperature_scale","ramp_seconds","bath_drop_kelvin","top_flux_watts_per_square_metre"):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError(f"{key} must be finite and positive")
        if len(self.h_bounds)!=2 or not all(math.isfinite(v) for v in self.h_bounds) or not math.isfinite(self.initial_h) or not 0<self.h_bounds[0]<self.initial_h<self.h_bounds[1]:
            raise ValueError("initial h must be inside positive bounds")
        object.__setattr__(self,"h_bounds",tuple(self.h_bounds))


CONFIG = asdict(PlanarRevisionConfig())


def _mlp(inputs,outputs,c):
    sizes=(inputs,)+(c.width,)*c.layers+(outputs,)
    layers=[]
    for i,(a,b) in enumerate(zip(sizes[:-1],sizes[1:])):
        layer=nn.Linear(a,b,dtype=torch.float64)
        nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i<len(sizes)-2: layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class PlanarRevisionPINN(nn.Module):
    def __init__(self,physics=PlanarContactConfig(),config=PlanarRevisionConfig(),*,mode="known_contact",known_h=None):
        super().__init__()
        if mode not in ("known_contact","inverse"):
            raise ValueError("revision supports explicit known-h or inverse modes")
        if (mode=="known_contact") != (known_h is not None):
            raise ValueError("known h is required only in the explicit forward diagnostic")
        if known_h is not None and not config.h_bounds[0]<=known_h<=config.h_bounds[1]:
            raise ValueError("known h lies outside bounds")
        self.physics,self.config,self.mode,self.known_h=physics,config,mode,known_h
        lo,hi=config.h_bounds; probability=(config.initial_h-lo)/(hi-lo)
        self.raw_h=nn.Parameter(torch.tensor(math.log(probability/(1-probability)),dtype=torch.float64),requires_grad=mode=="inverse")
        self.regions=[]
        if physics.contact_start>0:self.regions.append((0.,physics.contact_start,False))
        self.regions.append((physics.contact_start,physics.contact_end,True))
        if physics.contact_end<physics.width:self.regions.append((physics.contact_end,physics.width,False))
        self.networks=nn.ModuleList([_mlp(3,1,config)] if config.architecture=="soft_mlp"
                                    else [_mlp(3,2,config) for _ in self.regions])

    @property
    def contact(self):
        if self.mode=="known_contact": return torch.tensor(float(self.known_h),dtype=torch.float64)
        lo,hi=self.config.h_bounds
        return lo+(hi-lo)*torch.sigmoid(self.raw_h)

    def set_contact(self,h):
        if self.mode!="inverse": raise ValueError("cannot change an oracle contact")
        lo,hi=self.config.h_bounds
        if not lo<float(h)<hi:raise ValueError("updated h must be strictly inside bounds")
        probability=(float(h)-lo)/(hi-lo)
        with torch.no_grad():self.raw_h.fill_(math.log(probability/(1-probability)))

    def _region(self,q,index):
        p,c=self.physics,self.config
        a,b,contact=self.regions[index]
        x=q[:,0:1]*p.width
        u=(x-a)/(b-a)
        # Exterior side insulation is hard only at genuine exterior faces.
        if a==0 and b==p.width: feature=torch.cos(math.pi*u)
        elif a==0: feature=2*u*u-1
        elif b==p.width: feature=1-2*(1-u)**2
        else: feature=2*u-1
        # A is a bottom trace depending on x,t only. B's top-flat z input
        # gives B_z=0 at the top; its smoothstep factor may then retain
        # a free top value without spoiling the prescribed top derivative.
        envelope,time_feature=self._time_features(q[:,2:3])
        inp=torch.cat((feature,1-2*(1-q[:,1:2])**2,time_feature),dim=1)
        outputs=self.networks[index](inp)
        trace_input=torch.cat((feature,-torch.ones_like(q[:,1:2]),time_feature),dim=1)
        trace=self.networks[index](trace_input)[:,0:1]
        z=q[:,1:2];time=q[:,2:3]
        ramp=1-torch.exp(-time*c.duration/c.ramp_seconds)
        flux=c.top_flux_watts_per_square_metre*ramp
        bath=-c.bath_drop_kelvin*ramp if contact else torch.zeros_like(time)
        bi=self.contact*p.thickness/p.thermal_conductivity if contact else torch.zeros((),dtype=torch.float64)
        a_field=c.temperature_scale*envelope*trace
        b_field=c.temperature_scale*envelope*outputs[:,1:2]
        # phi'(0)=Bi*phi(0), phi'(1)=0. Correction has zero value
        # and normal derivative at bottom, free value at top, and zero
        # normal derivative at top (both smoothstep' and B_z vanish).
        phi=1+bi*(z-z*z/2)
        result=bath+a_field*phi+flux*p.thickness/(2*p.thermal_conductivity)*z*z
        result=result+z*z*(3-2*z)*b_field
        return result/c.temperature_scale

    def dimensionless_temperature(self,q,*,region=None):
        if self.config.architecture=="soft_mlp":
            envelope,time_feature=self._time_features(q[:,2:3])
            return envelope*self.networks[0](torch.cat((2*q[:,:2]-1,time_feature),dim=1))
        if region is not None:return self._region(q,region)
        # Evaluate only points assigned to each smooth region.  The vertical
        # interfaces have separate left/right diagnostic evaluations.
        result=torch.zeros_like(q[:,0:1])
        x=q[:,0]*self.physics.width
        for i,(a,b,_) in enumerate(self.regions):
            mask=(x>=a)&((x<=b) if i==len(self.regions)-1 else (x<b))
            if bool(mask.any()):result[mask]=self._region(q[mask],i)
        return result

    def _time_features(self,time):
        c=self.config
        envelope=time if c.temporal_envelope=="linear" else 1-torch.exp(-time*c.duration/c.ramp_seconds)
        if c.time_feature=="linear":return envelope,2*time-1
        diffusion_time=self.physics.thickness**2/self.physics.diffusivity
        feature=torch.log1p(time*c.duration/diffusion_time)/math.log1p(c.duration/diffusion_time)
        return envelope,2*feature-1

    def forward(self,coordinates):
        scales=torch.tensor([self.physics.width,self.physics.thickness,self.config.duration],dtype=torch.float64)
        return self.config.initial_temperature+self.config.temperature_scale*self.dimensionless_temperature(coordinates/scales)[:,0]


def _grad(value,q):
    return torch.autograd.grad(value,q,torch.ones_like(value),create_graph=True)[0]


def _times(rng,n):
    # Four equally allocated time strata; include the fast through-thickness
    # transient without changing the physical time window or residual scales.
    edges=np.array([.001,.033333333333,.1,.333333333333,1.])
    group=np.arange(n)%4;rng.shuffle(group)
    return edges[group]+rng.random(n)*(edges[group+1]-edges[group])


def revision_points(model,seed,*,independent=False):
    c,p=model.config,model.physics;rng=np.random.default_rng(seed)
    n=c.interior_points*(2 if independent else 1)
    q=rng.uniform(0,1,(n,3));q[:,2]=_times(rng,n)
    result={"interior":q}
    for name,axis,value in (("left",0,0.),("right",0,1.),("top",1,1.)):
        n=c.boundary_points*(2 if independent else 1)
        q=rng.uniform(0,1,(n,3));q[:,axis]=value;q[:,2]=_times(rng,n);result[name]=q
    for name,intervals in (("bottom_contact",[(p.contact_start,p.contact_end)]),
                            ("bottom_insulated",[(0.,p.contact_start),(p.contact_end,p.width)])):
        intervals=[(a,b) for a,b in intervals if b>a]
        if not intervals:continue
        n=c.boundary_points*(2 if independent else 1)
        q=rng.uniform(0,1,(n,3));q[:,1]=0;q[:,2]=_times(rng,n)
        lengths=np.array([b-a for a,b in intervals]);choice=rng.choice(len(intervals),n,p=lengths/lengths.sum())
        for j,(a,b) in enumerate(intervals):
            mask=choice==j;q[mask,0]=(a+(b-a)*rng.uniform(.0001,.9999,mask.sum()))/p.width
        result[name]=q
    if c.architecture=="hard_boundary_decomposed":
        for i in range(len(model.regions)-1):
            n=c.interface_points*(2 if independent else 1)
            q=rng.uniform(0,1,(n,3));q[:,0]=model.regions[i][1]/p.width
            q[:,1]=rng.uniform(.002,.998,n);q[:,2]=_times(rng,n)
            result[f"interface_{i}"]=q
    return result


def _derived_seed(seed,phase,index):
    return int.from_bytes(hashlib.sha256(f"planar-revision/{seed}/{phase}/{index}".encode()).digest()[:4],"big")


def _expected_families(physics,config):
    result=["interior","left","right","top","bottom_contact"]
    if physics.contact_start>0 or physics.contact_end<physics.width:result.append("bottom_insulated")
    if config.architecture=="hard_boundary_decomposed":
        count=int(physics.contact_start>0)+int(physics.contact_end<physics.width)
        for i in range(count):result.extend((f"interface_{i}_temperature",f"interface_{i}_flux"))
    return result


def revision_residuals(model,points):
    p,c=model.physics,model.config;result={}
    for name,values in points.items():
        q=torch.tensor(values,dtype=torch.float64,requires_grad=True)
        if name.startswith("interface_"):
            i=int(name.split("_")[1]);left=model.dimensionless_temperature(q,region=i);right=model.dimensionless_temperature(q,region=i+1)
            result[name+"_temperature"]=(left-right)*c.temperature_scale/c.interface_temperature_scale
            result[name+"_flux"]=p.thermal_conductivity*c.temperature_scale/p.width*(_grad(left,q)[:,0:1]-_grad(right,q)[:,0:1])/c.top_flux_watts_per_square_metre
            continue
        theta=model.dimensionless_temperature(q);g=_grad(theta,q)
        if name=="interior":
            xx=_grad(g[:,0:1],q)[:,0:1];zz=_grad(g[:,1:2],q)[:,1:2]
            result[name]=g[:,2:3]-c.duration*p.diffusivity*(xx/p.width**2+zz/p.thickness**2)
        elif name in ("left","right"):
            result[name]=p.thermal_conductivity*c.temperature_scale/p.width*g[:,0:1]/c.top_flux_watts_per_square_metre
        else:
            flux=p.thermal_conductivity*c.temperature_scale/p.thickness*g[:,1:2]
            ramp=1-torch.exp(-q[:,2:3]*c.duration/c.ramp_seconds)
            if name=="top":flux=flux-c.top_flux_watts_per_square_metre*ramp
            elif name=="bottom_contact":flux=flux-model.contact*(c.temperature_scale*theta+c.bath_drop_kelvin*ramp)
            result[name]=flux/c.top_flux_watts_per_square_metre
    return result


def diagnose_planar_revision(model,*,independent_seed=987654321):
    p,c=model.physics,model.config;rng=np.random.default_rng(independent_seed)
    x=(np.arange(40)+rng.uniform(.3,.7))/40;z=(np.arange(12)+rng.uniform(.3,.7))/12
    times=np.array([.25,.75,1.5,3.,6.,12.,22.,38.,60.,88.,115.])/c.duration
    xx,zz,tt=np.meshgrid(x,z,times,indexing="ij")
    points={"interior":np.column_stack((xx.ravel(),zz.ravel(),tt.ravel()))}
    xb=(np.arange(160)+rng.uniform(.3,.7))/160
    xx,tt=np.meshgrid(xb,times,indexing="ij");flatx=xx.ravel();flatt=tt.ravel()
    points["top"]=np.column_stack((flatx,np.ones(len(flatx)),flatt))
    contact=(flatx*p.width>=p.contact_start)&(flatx*p.width<=p.contact_end)
    points["bottom_contact"]=np.column_stack((flatx[contact],np.zeros(contact.sum()),flatt[contact]))
    if (~contact).any():points["bottom_insulated"]=np.column_stack((flatx[~contact],np.zeros((~contact).sum()),flatt[~contact]))
    zz,tt=np.meshgrid((np.arange(24)+.5)/24,times,indexing="ij")
    for side,boundary in (("left",0.),("right",1.)):
        points[side]=np.column_stack((np.full(zz.size,boundary),zz.ravel(),tt.ravel()))
    if c.architecture=="hard_boundary_decomposed":
        for i in range(len(model.regions)-1):
            points[f"interface_{i}"]=np.column_stack((np.full(zz.size,model.regions[i][1]/p.width),zz.ravel(),tt.ravel()))
    sums={};counts={}
    for name,coordinates in points.items():
        for chunk in range(0,len(coordinates),384):
            for key,value in revision_residuals(model,{name:coordinates[chunk:chunk+384]}).items():
                sums[key]=sums.get(key,0.)+float(torch.sum(value*value).detach());counts[key]=counts.get(key,0)+value.numel()
    values={key:math.sqrt(value/counts[key]) for key,value in sums.items()}
    scales={key:(c.temperature_scale/c.duration if key=="interior" else c.interface_temperature_scale if key.endswith("_temperature") else c.top_flux_watts_per_square_metre) for key in values}
    return {"normalized_rms":values,"physical_rms":{k:v*scales[k] for k,v in values.items()},"normalization_scales":scales,
            "point_counts":{k:len(v) for k,v in points.items()},"seed":independent_seed,
            "expected_families":_expected_families(model.physics,model.config),
            "validation_scheme":"independent shifted structured spatial grids and fixed explicit early/late times; chunked AD strong residuals",
            "times_s":(times*c.duration).tolist(),
            "all_families_pass":all(v<=.05 for v in values.values())}


def gate_planar_revision(report,field_metrics):
    values=report["independent_residuals"]["normalized_rms"]
    expected=_expected_families(PlanarContactConfig(**report["physics"]),PlanarRevisionConfig(**report["config"]))
    gates={"observations":report["normalized_observation_mse"]<=2,
           "each_independent_residual":bool(set(values)==set(expected) and all(math.isfinite(v) and v<=.05 for v in values.values())),
           "field":field_metrics["rmse_K"]<=.05 and field_metrics["maximum_error_K"]<=.15}
    gates["all"]=all(gates.values());return gates


def predict_planar_revision(model,coordinates):
    with torch.no_grad():return model(torch.tensor(np.asarray(coordinates),dtype=torch.float64)).numpy()


def train_planar_revision(observations,*,physics=PlanarContactConfig(),config=PlanarRevisionConfig(),mode="known_contact",known_h=None,
                          neural_seed=1,sampling_seed=2,independent_seed=3,initial_model=None,freeze_contact=True,progress=None):
    if not isinstance(observations,PlanarObservations):raise TypeError("only observed point temperatures may train this field")
    coordinates=observations.coordinates
    if (coordinates<0).any() or (coordinates[:,0]>physics.width).any() or (coordinates[:,1]>physics.thickness).any() or (coordinates[:,2]>config.duration).any():
        raise ValueError("observations lie outside the physical space-time domain")
    if sampling_seed==independent_seed:raise ValueError("independent diagnostic seed must differ")
    torch.set_num_threads(1);torch.manual_seed(neural_seed)
    model=initial_model if initial_model is not None else PlanarRevisionPINN(physics,config,mode=mode,known_h=known_h)
    if model.mode!=mode or (mode=="inverse" and known_h is not None):raise ValueError("warmstart mode mismatch or inverse truth leak")
    if model.physics!=physics or any(getattr(model.config,key)!=getattr(config,key) for key in ("architecture","width","layers","duration","initial_temperature","temperature_scale","ramp_seconds","bath_drop_kelvin","top_flux_watts_per_square_metre","h_bounds","temporal_envelope","time_feature")):
        raise ValueError("warmstart geometry, architecture or physical trace mismatch")
    if mode=="known_contact" and model.known_h!=known_h:raise ValueError("oracle warmstart h mismatch")
    model.config=config
    model.raw_h.requires_grad_(mode=="inverse" and not freeze_contact)
    parameters=[p for p in model.parameters() if p.requires_grad]
    obsq=torch.tensor(observations.coordinates.copy(),dtype=torch.float64);obst=torch.tensor(observations.temperature.copy(),dtype=torch.float64)
    seed_schedule=[_derived_seed(sampling_seed,"adam",i) for i in range((config.adam_epochs-1)//config.refresh_adam_every+1)]
    lbfgs_seed=_derived_seed(sampling_seed,"lbfgs",0)
    if independent_seed in seed_schedule+[lbfgs_seed] or len(set(seed_schedule+[lbfgs_seed]))!=len(seed_schedule)+1:
        raise ValueError("derived training/diagnostic seed collision")
    history=[];started=perf_counter();points=revision_points(model,seed_schedule[0])
    def loss():
        observation=torch.mean(((model(obsq)-obst)/observations.standard_deviation)**2)
        training_points=({k:v for k,v in points.items() if k=="interior" or k.startswith("interface_")}
                         if config.architecture=="hard_boundary_decomposed" and config.skip_hard_bc_training else points)
        residuals=revision_residuals(model,training_points)
        physics_loss=sum(torch.mean(v*v) for v in residuals.values())
        total=config.observation_weight*observation+config.physics_weight*physics_loss
        if not torch.isfinite(total):raise FloatingPointError("nonfinite revised PINN objective")
        return total,observation,physics_loss
    adam=torch.optim.Adam(parameters,lr=config.learning_rate)
    for epoch in range(config.adam_epochs):
        if epoch%config.refresh_adam_every==0:points=revision_points(model,seed_schedule[epoch//config.refresh_adam_every])
        adam.zero_grad();total,obs,phy=loss();total.backward();adam.step()
        if epoch%100==0 or epoch+1==config.adam_epochs:
            event={"stage":"adam","step":epoch+1,"objective_before_update":float(total.detach()),"observation":float(obs.detach()),"physics":float(phy.detach())}
            history.append(event)
            if progress:progress(event)
    points=revision_points(model,lbfgs_seed)
    actual=0;evaluations=0
    if config.lbfgs_iterations:
        opt=torch.optim.LBFGS(parameters,lr=1.,max_iter=config.lbfgs_iterations,max_eval=config.lbfgs_iterations*2,history_size=100,line_search_fn="strong_wolfe",tolerance_grad=1e-9,tolerance_change=1e-12)
        def closure():
            nonlocal evaluations
            opt.zero_grad();total,obs,phy=loss();total.backward();evaluations+=1
            if evaluations%100==0:
                event={"stage":"lbfgs_closure","step":evaluations,"objective_before_update":float(total.detach()),"observation":float(obs.detach()),"physics":float(phy.detach())};history.append(event)
                if progress:progress(event)
            return total
        opt.step(closure);actual=int(opt.state[parameters[0]].get("n_iter",0))
    total,obs,phy=loss()
    report={"architecture":config.architecture,"mode":mode,"known_h_oracle":mode=="known_contact","contact_estimate":float(model.contact.detach()),
            "contact_frozen":freeze_contact,"config":asdict(config),"physics":asdict(physics),"neural_seed":neural_seed,"sampling_seed":sampling_seed,
            "normalized_observation_mse":float(obs.detach()),"final_physics_loss":float(phy.detach()),"final_objective":float(total.detach()),
            "independent_residuals":diagnose_planar_revision(model,independent_seed=independent_seed),"lbfgs_actual_iterations":actual,"lbfgs_closure_evaluations":evaluations,
            "history":history,"wall_seconds":perf_counter()-started,"checkpoint":"optimizer-returned final state","hidden_truth_labels_used":False}
    report["derived_training_seeds"]={"adam":seed_schedule,"lbfgs":lbfgs_seed,"independent":independent_seed}
    report["identically_satisfied_training_families_omitted"]=(
        [k for k in ("left","right","top","bottom_contact","bottom_insulated") if k in points]
        if config.architecture=="hard_boundary_decomposed" and config.skip_hard_bc_training else [])
    return model,report
