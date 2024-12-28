#! /bin/bash
find images -maxdepth 2 -type f -name "*.jpg" ! -name "_SELFIE*.jpg" ! -path "images/@all_photos/*" -exec mv {} "images/@all_photos/" \;
