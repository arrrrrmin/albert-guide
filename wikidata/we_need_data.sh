# Requires that INSTALL was followed beforehand

if [ -z "$1" ]; then
  echo "langcode is unset"; else echo "var is set to '$1'";
fi

langcode=$1

wikidata_dump="${langcode}wiki-latest-pages-articles.xml.bz2"
echo "Downloading $wikidata_dump ..."

data="data/"

if [ -f $wikidata_dump ]; then
  cd ../wikiextractor
  python3 WikiExtractor.py  -q -b 55000K --section --min_text_length 300 --no_templates --json \
                            -o "../wikidata/${data}" "../wikidata/${wikidata_dump}"
else
  wget "http://download.wikimedia.org/${langcode}wiki/latest/${wikidata_dump}"
  cd ../wikiextractor
  python3 WikiExtractor.py  -q -b 55000K --section --min_text_length 300 --no_templates --json \
                            -o "../wikidata/${data}" "../wikidata/${wikidata_dump}"
fi
