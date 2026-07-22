#!/bin/bash
# In-process multithreaded CPU burn (models a hidden thread pool), then a
# valid tree from the baseline binary.
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/burner"
/Users/liujinguo/rcode/omeco/research/validator/bin/baseline_attempt "$1" "$2" "$3"
