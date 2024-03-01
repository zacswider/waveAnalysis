#!/bin/bash

# Check if watchmedo is installed
if ! command -v watchmedo &> /dev/null
then
    echo "watchmedo could not be found, please install it using 'pip install watchdog'"
    exit 1
fi

echo "Watching for file changes. Running pytest on file save..."
watchmedo shell-command \
    --patterns="*.py" \
    --recursive \
    --command='echo "Changes detected. Running pytest..."; pytest -W ignore:np.find_common_type is deprecated:DeprecationWarning; echo "Waiting for changes..."' \
    .
