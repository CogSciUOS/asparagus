# How to contribute to our report using latex

## Installation

Install (La)Tex:

- Linux: Tex Live
- Mac: Mac Tex
- Windows: MiKTeX

Just google or have a look [https://www.latex-tutorial.com/installation/](here).


## Generate a pdf output

Run `latexmk -pdf` in the report folder.

The blank pages in the output pdf are correct, as the reported will be printed one-sided.


## Tutorial

I do not find the Tutorial that I used to learn LaTex, but (this)[https://www.latex-tutorial.com/tutorials/] seems to give a good overview.


## Spaces

Use `~` to produce non-breakable space.


## Citations

Use `~\citep{BIBKEY}` to add a citation to the report. Make sure that you added the source with the corresponding BIBKEY to the `bibliography.bib` file.

It is mandatory that all bib entries are complete. Especially make sure that you include a **doi**!


## Figures

Add figures (make sure that the quality of the picture is good; .svg is preferable) to the `Figures` folders. In the report add:

```
\begin{figure}[ht]
	\centering
	\includegraphics[scale=0.4]{Figures/compare_raw_signal}
	\decoRule
	\caption[Signal of both eye-trackers]{The experiment consisted of six identical blocks. Each block starts with calibration phase and is followed by a fixed sequence of the ten conditions. Thus, each participant took part in six calibration procedures and a total of 60 conditions.}
	\label{fig:RawSignal}
\end{figure}
```

Write something **meaningful** as the label. You can reference it with `Figure~\ref{fig:RawSignal}`.



## Pushing to github

Run `latexmk -c` to clean up (delete auxiliary files).

Do **not** push the pdf to github (also no .aux files etc.). This is not necessary! Everyone can create a local pdf.


## main.tex

You probably do not need to change anything in the `main.tex` file except adding a new section file (that you put into the `Chapters` folder) to it:

```
% Include the chapters of the thesis as separate files from the Chapters folder

\include{Chapters/Introduction}
\include{Chapters/YourNewChapter}
```
