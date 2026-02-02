from typing import Dict, List, Any
from collections import defaultdict
import math
import time

RECENCY_HALFLIFE_YEARS = 8
MIN_DOCS_FOR_CONFIDENCE = 2
MAX_SCORE = 1.0