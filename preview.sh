#!/usr/bin/env bash
rm -rf posts/**/index.html index.html post_links.html .new_post_flag
act push --bind --env PREVIEW=true