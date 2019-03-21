# HOME
**HO**pefully-s**M**art n**E**ws aggregator

RSS Feed news aggregator + machinelearning 

## Learning phase/Building the initial database
RSS => Database => Learning => Manual dataset preparation

## Operational phase
RSS => | MODEL | => | DB | => | USER | => | DB |

1) RSS files are parsed by the python server
2) The datastructure is given as input for the machinelearning model
3) The output is stored to the database (with metadata)
4) First T results (chronological order) are provided to the user
----
5) User Likes/Dislikes/Read news (Update the database with Like/Dislike/Read)

## DB Structure

* IdNews (not used for the model, just to keep track of the likes/dislikes. it's a sha1(title|source|datetime) (i know it's bad, butwhocares))
* Title
* Digest
* URL
* Author
* Datetime
* Source
* Category 
* Like
* Dislike
* Read

## Features for learning

* Title
* Author
* Category
* Like
* Dislike
* Read

{...}

