#!/bin/bash
# For every directory under the given root, list at most 10 items.
root="/home/dbcloud/桌面/wound_segmentation_classification_processed"
find "$root" -type d | while read dir; do
    echo "$dir:"
    ls -1 "$dir" | head -n 10
    echo "----"
done
