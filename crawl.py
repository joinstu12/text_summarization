#coding = utf-8
import random
import socket
try:
    import urllib.request as urllib2
except:
    import urllib2
#import http.cookiejar as cookielib
import cookielib

ERROR = {
        '0':'Can not open the url,checck you net',
        '1':'Creat download dir error',
        '2':'The image links is empty',
        '3':'Download faild',
        '4':'Build soup error,the html is empty',
        '5':'Can not save the image to your disk',
    }

class BrowserBase(object): 

    def __init__(self):
        socket.setdefaulttimeout(20)

    def speak(self,name,content=None):
        print('[%s]%s' %(name,content))

    def openurl(self,url):
        """
       
        """
        cookie_support= urllib2.HTTPCookieProcessor(cookielib.CookieJar())
        self.opener = urllib2.build_opener(cookie_support,urllib2.HTTPHandler)
        urllib2.install_opener(self.opener)
        user_agents = [
                    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
                    'Opera/9.25 (Windows NT 5.1; U; en)',
                    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
                    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
                    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
                    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
                    "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
                    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 ",

                    ] 
       
        agent = random.choice(user_agents)
        self.opener.addheaders = [("User-agent",agent),("Accept","*/*"),('Referer','http://www.yahoo.com')]
        try:
            res = self.opener.open(url)
	    #return res
            #print res.read()
        except Exception as e:
            self.speak(str(e))
            raise Exception
        else:
            return res.read()

