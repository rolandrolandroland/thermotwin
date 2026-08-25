"""Public facade for the electrical-contact process-window study."""

from .design.contact_process_window import *  # noqa: F401,F403


def main(argv=None) -> None:
    from .reports.contact_process_window import main as report_main

    report_main(argv)


if __name__ == "__main__":
    main()
