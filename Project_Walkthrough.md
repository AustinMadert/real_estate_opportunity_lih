# Project Overview and Walkthrough

My intentions in starting this project were to use data to address the homelessness problem that many cities grapple with.
I currently live in Austin, TX and it is an example of a rapidly growing metropolis with increasing cost of living and
housing prices. This has a direct impact on homelessness populations. If you live in the area or have ever driven through
it's difficult to miss the panhandlers and vagrants on most major streets throughout the day.

My approach was to see if I could use the data to improve city planning and policy efforts, which I deemed to be a potentially
impactful approach. Austin currently has low income housing programs and I attempted to use geospatial data to improve
planning for these projects. In particular, I sought to address the question: "Can I identify the best houses to buy
within the city of Austin for a low income housing program?"

To address the problem I needed to acquire local property data and engineer features that would be relevant to a program.
Finally, I plotted the best houses and used visualization to help understand how those were generated using my model.


## Data acquisition
- using scrapy
    - link the trulia scrapy repo
    - link the time delay repo
- selenium
- full address list and lat longs
    - boosted regression tree

## Feature engineering
- distance to transportation
- clustering
- price per sqft

## Scoring and plotting