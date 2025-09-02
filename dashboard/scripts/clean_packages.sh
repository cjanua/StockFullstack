#!/usr/bin/env bash

npm install --save-dev depcheck

depcheck . | grep "^\* " | cut -d' ' -f2 | xargs npm uninstall

npm prune