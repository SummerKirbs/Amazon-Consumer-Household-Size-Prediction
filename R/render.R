# Rebuild the analysis and every figure the site uses.
#
# 1. Put survey_train_test.csv and amazon-purchases.csv in DATA/
# 2. Run this from the repository root:  Rscript R/render.R
#
# Why two steps: rmarkdown::render() only evaluates R chunks in .Rmd/.rmd files.
# Handed a .md file it goes straight to pandoc and the code blocks are left as text.
# So we knit first, then convert the knitted markdown to HTML.

if (!file.exists("model.md")) stop("Run this from the repository root.")

knitr::knit("model.md", output = "model.knit.md", quiet = FALSE)

rmarkdown::render(
  input       = "model.knit.md",
  output_file = "model.html",
  quiet       = TRUE
)

unlink("model.knit.md")

message("Rendered model.html and refreshed assets/img/")
