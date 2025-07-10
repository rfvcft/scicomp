#	Corporate Design 2016 Presentation

Last update 21.12.2017 by Mirco Altenbernd, Dominik Göddeke

For questions or bug reports
mailto: `mirco.altenbernd@ians.uni-stuttgart.de`

compiled and tested with pdflatex 3.14159265-2.6-1.40.15

## Usage

This is a beamer template for the new corporate design. To use it
within your beamer file just insert
	`\usetheme[simtech,german]{fbm2016}`
with optional options in squared brackets:
* `simtech`: 		enables the SimTech-Logo in the bottom center in the default footer
* `german`: 		switching to german, in particular: use the german uni logo
* `numbered`: 	shows frame number on bottom right (not on title page)

## Frontpage

There are the usual beamer and some additional commands to define the front page.

* `\title{}` 							(necessary)
	Sets the bold font headline in the big anthracite circle
*	`\subtitle{}`						(necessary)
	Sets the subtitle in normal font in the big anthracite circle
*	`\author{}`							(necessary)
	Sets the authors name inside the small light-blue circle.
*	`\setLogoText{}`				(optional)
	Sets a specific text below the logo of the university on the front page.
	In the english version this means that 'Germany' is replaced.
*	`\date{}`								(optional)
	Specify the date on the front page in the big anthracite circle
*	`\setTitlePic{}`				(optional)
	Set the front page picture, with default format (as titlepic.jpg). Default: `\setTitlepic{titlepic.jpg}`
*	`\defTitlePic{}`				(optional)
	More flexible version of '\setTitlePic'. Can also define resolutions. Default: `\defTitlepic{\includegraphics[height=6cm]{titlepic.jpg}}`
	If used with an empty input there is no titlepic used.

## Content

The content is as usual wrapped inside of the document environment

	\begin{document}
		Content
	\end{document}

and then defined frame by frame

	\begin{frame}{frametitle}
		Content
	\end{frame}

In addition there is a `dualframe`-environment to use frames with two columns:

	\begin{dualframe}[ratio]{frametitle}
	{
		left content
	}{
		right content
	}
	\end{dualframe}

where `ratio` defines the ratio between both columns (0.5 per default). The additional brackets are necessary.

The default content font size is `\normalsize`. To structure the content
you can use
	`\section{}`
and
	`\subsection{}`
as in common latex documents, and create a TOC via `\tableofcontents`.

## Footer

A footer is shown on every slide. The command `\toggleFooter` disables/enables the footer on the following slides.

By default there is a footer with the file `logo_institute.pdf` at the bottom left. If the option
`simtech` is used, `logo_simtech.pdf` is at the bottom mid and `logo_institute.pdf` is at the bottom
left. In addition there is the uni logo on all frames except the front page. The option
	`\setFooter{}`
enables an individual footer, see the preamble of the demo slides for documentation.