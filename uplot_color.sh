#!/bin/bash
# duckdb $1 -s "COPY (select epoch, value from tensors where name='$2') TO '/dev/stdout' WITH (FORMAT 'csv', HEADER)" | uplot density -d, -H --width 150
duckdb $1 -s "COPY (select epoch, value from tensors where name='$2' and value > 0) TO '/dev/stdout' WITH (FORMAT 'csv', HEADER)" | uplot density -d, -H --width 150
# duckdb duck.db -s "COPY (select value from tensors) TO '/dev/stdout' WITH (FORMAT 'csv', HEADER)" | uplot hist -d, -H --nbins 50

