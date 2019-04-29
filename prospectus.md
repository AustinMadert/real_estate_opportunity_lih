# Identifying Real Estate Opportunity for Low Income Projects

## What am I trying to do?

The purpose of this project is to identify properties within and around the city of Austin that would represent the best investments for low income housing initiatives. It would seek to identify properties that would minimize the potential cost to the city, place individuals near important amenities (such as public transportation), and maintain dispersion of properties through different areas of the city and surrounding areas.

Specifically, the project would seek to answer the questions: 
    - Given a specific budget, what are the top property options available in Austin?
    - Given a specific number of tenants, what are the top property options available in Austin?
    - Given a specific budget and a specific number of potential tenants, what are the top property options available in Austin?


## How has this problem been approached before?

Since studies in public housing have shown that large centralized complexes known as housing projects have led to increased crime, drug use,
lower performance in schools, a more decentralized model has become common in cities. Modern public housing programs include properties spread across cities and surrounding areas to better assimilate the lower income tenants. 

Other projects have used machine learning to generate models that map rent decay from dense city centers with little available data. This project accomplished this using data from the metropolitan of Guangzhou, China: https://www.sciencedirect.com/science/article/abs/pii/S0143622816303204

There is also a more qualitative body of work that seeks to leverage learnings from city management learnings from around the world that form a strong foundation for how to improve models that I generate: http://housingsupply.our.dmu.ac.uk/files/2014/03/ESRC-Report-FINAL.pdf


## What is new about this approach?

The approach this project takes is to earmark certain properties for purchase or if need-be immenent domain. Austin is a rapidly growing city and housing prices are likely to continue rising rapidly which creates a burden on lower income households that depend on city employment. Having high value properties earmarked before development, or before they come on the market is of value because it steamlines the process and potentially saves the city money be correctly timing acquisitions and partnerships with developers. 

This project will attempt to approach the problem using clustering techniques to create target groups for acquisition.


## So what?

The potential impact would be to allow for more efficient use of local government funds. Making programs that are potentially expensive on this level are difficult to justify and minimizing the cost while maximizing their impact is important in proving the value to this initiative and future initiatives.


## How will this be presented?

I plan to create a slide presentation. 

A stretch goal is to perhaps leverage geospatial data from Open City Model to create rich visuals. 

An additional stretch goal would be to create a web application that would allow a user to interactively change their target budget and number of tenants to readjust the output top properties.

## Data Sources

I will be leveraging multiple sources of data for this project. I will use community data from the US census Bureau PUMS (Public Use Microdata Sample) datasets. The 2008-2010 PUMS represents in the ballpark of 10 GB of data. 

I will use Zillow data and sources from local government to collect housing prices to use to identify properties.

As a stretch goal, I will leverage the Open City Model project's 3D geospatial datasets to create rich visuals.


## What are the potential problems with the project?

Potential problems could include issues with getting data for properties within Austin. Zillow has listing prices and zestimates for many properties. The Zestimates are not guaranteed to be accurate and listing prices don't include properties not on the market. This might represent a problem of either incomplete data or the need to compute some estimate prices for properties where I can't get valuation data. 

Feature engineering might also be a complicated task. I want to include not only value, but proximity to amenities and dispersion in order to ensure the selections the model makes are useful within the context of housing projects and not simply pricing data. This means I will have to compute additionaly features to factor these aspects in and that could make for a lot of tinkering. I also think that it'll make the project more difficult to explain.

Getting meaningful results might be somewhat difficult. It might be hard to interpret the necessary course of action if for instance the model predicts the best value to come from as of yet undeveloped land. Perhaps I will need to create multiple models that deal with different classes of property.


## What is the next thing to work on?

I will need to finish collecting data. I need to finish acquiring property prices. I will also need to begin cleaning data and potentially computing price estimates for properties that I cannot get a hold of. From there I will begin feature engineering.