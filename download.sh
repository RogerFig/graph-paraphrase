#!/bin/bash

echo "Downloading Word2Vec skip_s50"
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s50.zip
mkdir -p "embeddings"
unzip skip_s50.zip -d /embeddings
rm skip_s50.zip
echo "Done!!!"

echo "Downloading Word2Vec skip_s100"
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s100.zip
mkdir -p "embeddings"
unzip skip_s100.zip -d /embeddings
rm skip_s100.zip
echo "Done!!!"

echo "Downloading Word2Vec skip_s300"
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s300.zip
mkdir -p "embeddings"
unzip skip_s300.zip -d /embeddings
rm skip_s300.zip
echo "Done!!!"

echo "Downloading FastText skip_s50"
wget http://143.107.183.175:22980/download.php?file=embeddings/fasttext/skip_s50.zip
mkdir -p "embeddings"
unzip skip_s50.zip -d /embeddings
rm skip_s50.zip
echo "Done!!!"

echo "Downloading FastText skip_s100"
wget http://143.107.183.175:22980/download.php?file=embeddings/fasttext/skip_s100.zip
mkdir -p "embeddings"
unzip skip_s100.zip -d /embeddings
rm skip_s100.zip
echo "Done!!!"

echo "Downloading FastText skip_s300"
wget http://143.107.183.175:22980/download.php?file=embeddings/fasttext/skip_s300.zip
mkdir -p "embeddings"
unzip skip_s300.zip -d /embeddings
rm skip_s300.zip
echo "Done!!!"

echo "Downloading Wang2Vec skip_s50"
wget http://143.107.183.175:22980/download.php?file=embeddings/wang2vec/skip_s50.zip
mkdir -p "embeddings"
unzip skip_s50.zip -d /embeddings
rm skip_s50.zip
echo "Done!!!"

echo "Downloading Wang2Vec skip_s100"
wget http://143.107.183.175:22980/download.php?file=embeddings/wang2vec/skip_s100.zip
mkdir -p "embeddings"
unzip skip_s100.zip -d /embeddings
rm skip_s100.zip
echo "Done!!!"

echo "Downloading Wang2Vec skip_s300"
wget http://143.107.183.175:22980/download.php?file=embeddings/wang2vec/skip_s300.zip
mkdir -p "embeddings"
unzip skip_s300.zip -d /embeddings
rm skip_s300.zip
echo "Done!!!"

echo "Downloading Glove 50"
wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s50.zip
mkdir -p "embeddings"
unzip glove_s50.zip -d /embeddings
rm glove_s50.zip
echo "Done!!!"

echo "Downloading Glove 100"
wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s100.zip
mkdir -p "embeddings"
unzip glove_s100.zip -d /embeddings
rm glove_s100.zip
echo "Done!!!"

echo "Downloading Glove 300"
wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip
mkdir -p "embeddings"
unzip glove_s300.zip -d /embeddings
rm glove_s300.zip
echo "Done!!!"