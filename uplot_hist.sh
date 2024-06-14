#!/bin/bash
duckdb $1 -s "COPY (select epoch, log(value) from grads where name='$2' and value > 0) TO '/dev/stdout' WITH (FORMAT 'csv', HEADER)" | uplot density -d, -H --width 150 --height 40 --grid
# duckdb duck.db -s "COPY (select epoch, log(value) from grads) TO '/dev/stdout' WITH (FORMAT 'csv', HEADER)" | uplot density -d, -H --nbins 50
