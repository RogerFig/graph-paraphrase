#!/bin/bash

mkdir -m 777 "embeddings"

echo "Downloading Word2Vec skip_s50"
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s50.zip -O skip_s50.zip
unzip skip_s50.zip -d embeddings && mv skip_s50.txt word2vec_50.txt
rm skip_s50.zip
echo "Done!!!"

echo "Downloading Word2Vec skip_s100"
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s100.zip -O skip_s100.zip
unzip skip_s100.zip -d embeddings && mv skip_s100.txt word2vec_100.txt
rm skip_s100.zip
echo "Done!!!"

echo "Downloading Word2Vec skip_s300"
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s300.zip -O skip_s300.zip
unzip skip_s300.zip -d embeddings && mv skip_s300.txt word2vec_300.txt
rm skip_s300.zip
echo "Done!!!"

echo "Downloading FastText skip_s50"
wget http://143.107.183.175:22980/download.php?file=embeddings/fasttext/skip_s50.zip -O skip_s50.zip
unzip skip_s50.zip -d embeddings && mv skip_s50.txt fasttext_50.txt
rm skip_s50.zip
echo "Done!!!"

echo "Downloading FastText skip_s100"
wget http://143.107.183.175:22980/download.php?file=embeddings/fasttext/skip_s100.zip -O skip_s100.zip
unzip skip_s100.zip -d embeddings && mv skip_s100.txt fasttext_100.txt
rm skip_s100.zip
echo "Done!!!"

echo "Downloading FastText skip_s300"
wget http://143.107.183.175:22980/download.php?file=embeddings/fasttext/skip_s300.zip -O skip_s300.zip
unzip skip_s300.zip -d embeddings && mv skip_s300.txt fasttext_300.txt
rm skip_s300.zip
echo "Done!!!"

echo "Downloading Wang2Vec skip_s50"
wget http://143.107.183.175:22980/download.php?file=embeddings/wang2vec/skip_s50.zip -O skip_s50.zip
unzip skip_s50.zip -d embeddings && mv skip_s50.txt wang2vec_50.txt
rm skip_s50.zip
echo "Done!!!"

echo "Downloading Wang2Vec skip_s100"
wget http://143.107.183.175:22980/download.php?file=embeddings/wang2vec/skip_s100.zip -O skip_s100.zip
unzip skip_s100.zip -d embeddings && mv skip_s100.txt wang2vec_100.txt
rm skip_s100.zip
echo "Done!!!"

echo "Downloading Wang2Vec skip_s300"
wget http://143.107.183.175:22980/download.php?file=embeddings/wang2vec/skip_s300.zip -O skip_s300.zip
unzip skip_s300.zip -d embeddings && mv skip_s300.txt wang2vec_300.txt
rm skip_s300.zip
echo "Done!!!"

echo "Downloading Glove 50"
wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s50.zip -O glove_s50.zip
unzip glove_s50.zip -d embeddings
rm glove_s50.zip
echo "Done!!!"

echo "Downloading Glove 100"
wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s100.zip -O glove_s100.zip
unzip glove_s100.zip -d embeddings
rm glove_s100.zip
echo "Done!!!"

echo "Downloading Glove 300"
wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip -O glove_s300.zip
unzip glove_s300.zip -d embeddings
rm glove_s300.zip
echo "Done!!!"