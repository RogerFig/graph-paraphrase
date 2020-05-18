#!/bin/bash

echo "Download skip_s50"
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s50.zip
mkdir -p "embeddings"
unzip skip_s50.zip -d /embeddings
rm skip_s50.zip

echo "Download skip_s100"
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s100.zip
mkdir -p "embeddings"
unzip skip_s100.zip -d /embeddings
rm skip_s100.zip

echo "Download skip_s300"
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s300.zip
mkdir -p "embeddings"
unzip skip_s300.zip -d /embeddings
rm skip_s300.zip

echo "Done!!"