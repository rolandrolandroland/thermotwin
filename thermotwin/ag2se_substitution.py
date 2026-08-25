"""Public facade for the matched Ag2Se substitution study."""

from .design.ag2se_substitution import *  # noqa: F401,F403


def main(argv=None) -> None:
    from .reports.ag2se_substitution import main as report_main

    report_main(argv)


if __name__ == "__main__":
    main()
