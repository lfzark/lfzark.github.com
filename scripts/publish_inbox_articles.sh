#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INBOX_DIR="$ROOT_DIR/article_inbox"
INDEX_FILE="$ROOT_DIR/index.html"
SITEMAP_FILE="$ROOT_DIR/sitemap.xml"
SITE_BASE="https://arkpy.com"

if [[ ! -d "$INBOX_DIR" ]]; then
  echo "Missing inbox dir: $INBOX_DIR" >&2
  exit 1
fi

if [[ ! -f "$INDEX_FILE" || ! -f "$SITEMAP_FILE" ]]; then
  echo "Missing index.html or sitemap.xml" >&2
  exit 1
fi

mapfile -t files < <(find "$INBOX_DIR" -maxdepth 1 -type f -name 'article-*.html' ! -name 'article-template.html' -printf '%f\n' | sort)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No article-*.html files found in article_inbox/."
  exit 0
fi

echo "Publishing ${#files[@]} article(s)..."

recent_items=()
for file in "${files[@]}"; do
  src="$INBOX_DIR/$file"
  dest="$ROOT_DIR/$file"

  cp "$src" "$dest"

  title=$(grep -o '<h1>.*</h1>' "$src" | head -n1 | sed -E 's/<[^>]+>//g' || true)
  if [[ -z "$title" ]]; then
    title="$file"
  fi

  today=$(date +%Y-%m-%d)
  recent_items+=("          <li>${today}：<a href=\"/${file}\">${title}</a></li>")

  url="${SITE_BASE}/${file}"
  if ! grep -q "$url" "$SITEMAP_FILE"; then
    tmp_sitemap=$(mktemp)
    awk -v u="$url" 'BEGIN{added=0} /<\/urlset>/{if(!added){print "  <url><loc>" u "</loc></url>"; added=1} } {print}' "$SITEMAP_FILE" > "$tmp_sitemap"
    mv "$tmp_sitemap" "$SITEMAP_FILE"
  fi

done

recent_block=$(printf '%s\n' "${recent_items[@]}")

tmp_index=$(mktemp)
awk -v block="$recent_block" '
  BEGIN {in_block=0}
  /AUTO_RECENT_START/ {
    print
    print block
    in_block=1
    next
  }
  /AUTO_RECENT_END/ {
    in_block=0
    print
    next
  }
  {
    if (!in_block) print
  }
' "$INDEX_FILE" > "$tmp_index"
mv "$tmp_index" "$INDEX_FILE"

echo "Done. Updated root articles, sitemap.xml, and index.html recent list."
