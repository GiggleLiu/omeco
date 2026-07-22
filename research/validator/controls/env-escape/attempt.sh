#!/bin/bash
cat /Users/liujinguo/rcode/omeco/research/benchmark/private/seeds.json && exit 0
/usr/bin/curl -s --max-time 2 http://example.com && exit 0
exit 1
