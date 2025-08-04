#!/bin/bash

# 阈值（单位 MB）
THRESHOLD=100

echo "🔍 Searching for files > ${THRESHOLD}MB..."

# 临时存储
TMPFILE=$(mktemp)

# 找出所有超过指定大小的文件
find . -type f -size +"${THRESHOLD}"M > "$TMPFILE"

# 加入 .gitignore
while IFS= read -r filepath; do
    # 去掉开头的 ./ 以便在 .gitignore 中更通用
    clean_path="${filepath#./}"
    # 检查 .gitignore 里是否已经存在
    if ! grep -Fxq "$clean_path" .gitignore 2>/dev/null; then
        echo "$clean_path" >> .gitignore
        echo "📝 Added to .gitignore: $clean_path"
    else
        echo "✅ Already ignored: $clean_path"
    fi
done < "$TMPFILE"

# 清理
rm "$TMPFILE"
echo "✅ Done. Check .gitignore!"
