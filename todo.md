# Promotion

1. Stackoverflow reply with link to post 🚀
2. Thank you reply to Chinese guy with link to post 🚀
3. Send to Alex Guzey 🚀

4. Find image to share
    - masked-array post -> plot of speed up
    - blog itself -> _________

4. Blog *Social Preview*
 - Images should be at least 640×320px (1280×640px for best display). 40pt border.
5. Image for post
 - There's a *Social Preview*?

6. Share
 - Twitter
 - LinkedIn
7. Put blog home page link on
 - Twitter
 - LinkedIn
 - CV


 # ZU

 1. Advisee
 2. Supervisee
 3. Neuro psychologist
 4. **Gaming manuscript**
 5. CAS files
 6. Syllabi



 # 
[Preview for social media](https://www.opengraph.xyz/)

https://quarto.org/docs/websites/website-tools.html#twitter-cards

https://quarto.org/docs/websites/website-tools.html#preview-images


# Cynthia Huang
https://www.cynthiahqy.com/posts/twitter-card-quarto/

## Image
Change tests-missing.svg to featured.png

1. make_featured.py temporary script to test
    - write to png
    - font sizes et cetera
        - `print(plt.style.available)`
        - `plt.style.use('ggplot')`
        - [gallery](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)
            - large font
            - `seaborn-poster`
            - `fivethirtyeight`
2. copy code into qmd and re-render post
    - copy **only fig format and export**
    - name as "featured.png"
    - **OR** just export *tests-missing.csv* in qmd for separate script

## Post
Add to index.qmd yml for masked-array post:

image: featured.png
image-alt: "Hand-drawn black and white wireframe sketch of a tweet containing a preview frame showing the text 'TWITTER CARD?!?'"
card-style: "summary"