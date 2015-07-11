import crawl
import re
from pyquery import PyQuery
def crawl_article(title):
	spider = crawl.BrowserBase()
	tmp = 'http://www.ted.com/talks/'+title +'/transcript'
	req = spider.openurl(tmp)
	pq = PyQuery(req)
	ti = pq('title')
	#print(type(ti))
	art = re.compile(r'\:.*?\|',re.S)
	art = re.search(art,ti.text()).group()
	punc = re.compile(r'\: | \|',re.S)
	art = punc.sub('',art)
	tag = pq('p.talk-transcript__para')
	time = re.compile(r'[0-9]+\:[0-9]+')
	tmp = time.sub('\@timestamp ',tag.text())
	tmp = tmp.split('\@timestamp ')
	sen = re.compile(r'.*?\.')
	for i in range(len(tmp)):
		tmp_sen = sen.finditer(tmp[i])
		tmp_i = []
		for j in tmp_sen:
			tmp_i.append(j.group())
		tmp[i] = tmp_i
	return tmp,art
crawl_article('mihaly_csikszentmihalyi_on_flow')
