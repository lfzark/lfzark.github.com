# 文章投放目录（自动发布）

把新文章 HTML 放到本目录，然后运行发布脚本即可自动更新站点。

## 命名规范

- 文件名必须用：`article-xxx.html`
- 示例：`article-xiaozhoutian-guide.html`

## 内容规范（必须）

- `<title>`：包含核心关键词
- `<meta name="description">`：60-90 字
- `<h1>`：文章主标题（脚本会读取这里显示到首页最近更新）
- 正文尽量超过 800 字，至少 2 个 `h2`

## 发布命令

在仓库根目录执行：

```bash
bash scripts/publish_inbox_articles.sh
```

## 脚本会自动做什么

1. 将 `article_inbox/article-*.html` 复制到站点根目录
2. 将页面 URL 写入 `sitemap.xml`
3. 更新 `index.html` 的 `最近更新` 自动区域（`AUTO_RECENT` 标记内）

## 注意

- 已发布文章会被覆盖更新（同名文件）
- 若要下线文章：删除根目录对应 HTML，并手动从 sitemap 与首页列表移除
