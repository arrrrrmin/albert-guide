# Requires that INSTALL was followed beforehand

wikidata_dump="itwiki-latest-pages-articles.xml.bz2"
extracted="extracted/"

# WikiExtractor.py --min_text_length --no-templates --json


if [ -f $wikidata_dump ]; then
  cd ../wikiextractor
  python3 WikiExtractor.py  -b 55000K --min_text_length 300 --no_templates --json \
                            -o "../wikidata/$extracted" "../wikidata/$wikidata_dump"
else
  wget http://download.wikimedia.org/itwiki/latest/$wikidata_dump
  cd ../../wikiextractor
  python3 WikiExtractor.py  -b 55000K --min_text_length 300 --no_templates --json \
                            -o "../wikidata/$extracted" "../wikidata/$wikidata_dump"
fi
# find $extracted -name '*bz2' -exec bzip2 -c {} \; > text.xml
# rm -rf $extracted