import argparse
import re
import sys

HEAD = """
<html>
<head>
  <script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: ["tex2jax.js", "TeX/AMSsymbols.js", "TeX/AMSmath.js"],
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      inlineMath: [ ['$','$'] ],
      displayMath: [ ['$$','$$'] ],
      processEscapes: true
    },
    "HTML-CSS": { availableFonts: ["TeX"] }
  });
  </script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js" async></script>
  <script type="text/javascript" src="../.tools/theme/marked.js">
  </script>
  <link href="http://cdn.bootcss.com/highlight.js/9.9.0/styles/darcula.min.css" rel="stylesheet">
  <script src="http://cdn.bootcss.com/highlight.js/9.9.0/highlight.min.js"></script>
  <link href="http://cdn.bootcss.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/perfect-scrollbar/0.6.14/css/perfect-scrollbar.min.css" rel="stylesheet">
  <link href="../.tools/theme/github-markdown.css" rel='stylesheet'>
</head>
<style type="text/css" >
.markdown-body {
    box-sizing: border-box;
    min-width: 200px;
    max-width: 980px;
    margin: 0 auto;
    padding: 45px;
}
</style>


<body>

<div id="context" class="container-fluid markdown-body">
</div>

<!-- This block will be replaced by each markdown file content. Please do not change lines below.-->
<div id="markdown" style='display:none'>
"""

TAIL = """
</div>
<!-- You can change the lines below now. -->

<script type="text/javascript">
marked.setOptions({
  renderer: new marked.Renderer(),
  gfm: true,
  breaks: false,
  smartypants: true,
  highlight: function(code, lang) {
    code = code.replace(/&amp;/g, "&")
    code = code.replace(/&gt;/g, ">")
    code = code.replace(/&lt;/g, "<")
    code = code.replace(/&nbsp;/g, " ")
    return hljs.highlightAuto(code, [lang]).value;
  }
});
document.getElementById("context").innerHTML = marked(
        document.getElementById("markdown").innerHTML)
</script>
</body>
"""


def convert_markdown_into_html(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*', help='Filenames to fix')
    args = parser.parse_args(argv)

    retv = 0

    for filename in args.filenames:
        with open(
                re.sub(r"README", "index", re.sub(r"\.md$", ".html", filename)),
                "w") as output:
            output.write(HEAD)
            with open(filename) as input:
                for line in input:
                    output.write(line)
            output.write(TAIL)

    return retv


if __name__ == '__main__':
    sys.exit(convert_markdown_into_html())
