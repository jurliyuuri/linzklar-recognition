# linzklar-recognition

## What this is 
Linzklar is a logographic writing system used mainly in [Faikleone](https://wikirlevip.miraheze.org/wiki/Faikleone), a [conworld](https://en.wikibooks.org/wiki/Conworld) the authors of this repository are building. 

![](DSC_1318-01.jpeg)

![](hsjoihs_.png)

![](DDf9lFVUwAAWe7c.jpg)

This repository currently contains the dataset of hand-written linzklar (written by the builders of the conworld through a [Web app](https://github.com/jurliyuuri/linzi-recognition)), with which we aim to implement a handwriting linzklar input system in the future.

## Data format

[/data](https://github.com/jurliyuuri/linzklar-recognition/tree/master/data) contains all the raw data, where each JSON file is either an array of `{ "character": (annotation), "data" : (array of array of {x: number, y: number}) }`, or `{ "character": (annotation), "data" : (array of array of {x: number, y: number}), initialDotCaptured: true}`. Each array of coordinates corresponds to a stroke; an array of strokes makes up a character. Due to an oversight, for the first 30,000 characters the initial dot of every single stroke is absent from the data, and such files are marked with the lack of `initialDotCaptured: true`.

## Shuffle through the dataset
[Shuffle through the dataset](http://jurliyuuri.com/linzklar-recognition/random.html).

## これはなに
燐字データセットのデータ集
 
cf. [thanks.txt](https://github.com/jurliyuuri/linzi-recognition/blob/master/thanks.txt)
