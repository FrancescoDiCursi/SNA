import twint
import nest_asyncio
import datetime

def collectData(since, until, namesToSearch, fileName):
    nest_asyncio.apply()
    for name in namesToSearch:
        c = twint.Config()
        c.Search = name
        c.Since = since
        c.Until = until
        c.Store_json = True
        c.Output = "data_collection/data/" + fileName + ".json"
        twint.run.Search(c)




