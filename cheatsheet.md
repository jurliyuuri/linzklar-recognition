## SVGを生成せずに字数カウントだけする

```sh
cd converter
cargo run --release -- --dry-run
```

## PNG と SVG を差分コンパイル

```sh
mv data data~
mv data_images data_images~
mv datalist.json datalist.json~
mkdir data
touch datalist.json
【Manually add the data to /data; write the file names to datalist.json】
cd converter
cargo run --release
cd ../data_images/
cp -r -v svg/ png
cd png
find . -name '*.svg' -exec mogrify -format png {} +
find . -name '*.svg' -type f -delete
cd ../../
cp -r data~/ data
rm -rf data~
cp -r data_images/ data_images~
rm -rf data_images
mv data_images~ data_images
【Manually merge datalist.json and datalist~.json】
```