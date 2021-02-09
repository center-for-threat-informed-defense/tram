import re

from app.objects import Search, Match


class RegexParser:

    @staticmethod
    async def find(regex, report, blob):
        try:
            for m in set([m for m in re.findall(regex['regex'], blob)]):
                report.matches.append(
                    Match(search=Search(tag='ioc', name=regex['name'], description=m, code=regex['code']))
                )
        except Exception:
            print("Something happened with one of the regexes...")
