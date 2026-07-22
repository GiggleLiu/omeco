#!/bin/bash
# Hidden in-process parallelism: 3 spin threads burn CPU for the entire run
# while the baseline binary does the real work single-threaded.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/burner2" /Users/liujinguo/rcode/omeco/research/validator/bin/baseline_attempt "$1" "$2" "$3"
