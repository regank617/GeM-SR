"""Thin CLI: Meta CHMv2 AWS fetch + ECC alignment + comparison plot.

Prerequisites: AWS CLI for Meta GeoTIFF download (``aws s3 cp --no-sign-request ...``).
"""

from scripts.run_meta_comparison import main

if __name__ == "__main__":
    main()
