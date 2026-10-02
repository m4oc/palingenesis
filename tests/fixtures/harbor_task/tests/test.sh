#!/usr/bin/env bash
mkdir -p /logs/verifier
if [ "$(cat /workdir/hello.txt 2>/dev/null | tr -d '[:space:]')" = "$EXPECTED" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
