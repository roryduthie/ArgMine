#!/bin/sh
source venv/bin/activate
exec gunicorn -b :8100 --access-logfile - --error-logfile - app --timeout 3000
