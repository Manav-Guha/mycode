"""Allow running myCode as: python -m mycode"""

import sys

from mycode.cli import main

sys.exit(main())
