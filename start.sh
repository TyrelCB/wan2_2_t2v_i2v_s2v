#!/bin/bash
set -e
python3 app.py &
python3 mcp_server.py &
wait -n
