# Wikipedia world-knowledge corpus attribution

The generated `world_wikipedia.jsonl` corpus is collected from the public Wikipedia API for Dori AI training and retrieval.

- Source: https://www.wikipedia.org/
- API: https://www.mediawiki.org/wiki/API:Main_page
- Text license: Wikipedia article text is generally available under Creative Commons Attribution-ShareAlike (CC BY-SA); individual pages may contain additional attribution requirements or exceptions.
- Each record retains its article title, source URL, language, retrieval timestamp, and license note.
- Before redistributing a model or corpus trained on this material, review the applicable license and attribution/share-alike obligations. This corpus is a limited, selected subset, not a complete copy of Wikipedia.

The collector intentionally uses a bounded list of topics and a small number of search results per topic. It is intended to broaden factual coverage, not to imply that a small model can learn all world knowledge or that every retrieved article is authoritative.
