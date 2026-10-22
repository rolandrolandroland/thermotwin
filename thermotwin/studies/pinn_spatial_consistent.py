"""Execute the separately declared top-derivative corrective development once."""
import argparse
import json

from .pinn_spatial_continuity import ForwardCallbacks, run_development


def consistent_callbacks():
    from ..pinn.planar_contact_consistent import (
        ConsistentConfig, fit_consistent, checkpoint_bytes,
    )
    from ..pinn.planar_contact_continuity import predict_continuity, diagnose_continuity

    def fit(observations, **kwargs):
        kwargs['config'] = ConsistentConfig(**kwargs['config'])
        return fit_consistent(observations, **kwargs)

    return ForwardCallbacks(fit=fit, predict=predict_continuity,
        diagnose=diagnose_continuity, checkpoint=checkpoint_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args(argv)
    result = run_development(args.output_directory, consistent_callbacks())
    print(json.dumps({'status': result['status'], 'case_count': len(result['cases']),
                      'wall_seconds': result['wall_seconds']}, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
