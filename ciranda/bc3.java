import ciranda.*;
import ciranda.features.*;
import ciranda.utils.*;
import ciranda.classify.*;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;

public class bc3{

public static void main(String[] args) throws IOException{

String msg = "hi jim, when are you coming back? how's the trip to Denmark going? pretty cold, huh? well, things here are going well, no problems so far. so...enjoy your trip, regards, jose"; 
for(int i = 0;i<40;i++){
FileReader fr = new FileReader("bc3_doc/mail_" + i);
FileWriter fw = new FileWriter("bc3_doc/mail_" + i + "_speech");
SpeechAct sa = new SpeechAct();
BufferedReader br = new BufferedReader(fr);
while (br.ready()) {
	String tmp = br.readLine();
	sa.loadMessage(tmp);
	//FileWriter fw = new FileWriter("bc3_doc/mail_" + i + "_speech");
	fw.write("Req = "+sa.hasRequest()+"  , Confidence= "+sa.getRequestConfidence() + "\n");
	fw.write("Dlv = "+sa.hasDeliver()+"  , Confidence= "+sa.getDeliverConfidence() + "\n");
	fw.write("Cmt = "+sa.hasCommit()+"  , Confidence= "+sa.getCommitConfidence() + "\n");
	fw.write("Prop = "+sa.hasPropose()+"  , Confidence= "+sa.getProposeConfidence() + "\n");
	fw.write("Meet = "+sa.hasMeet()+"  , Confidence= "+sa.getMeetConfidence() + "\n");
	fw.write("Ddata = "+sa.hasDdata()+"  , Confidence= "+sa.getDdataConfidence() + "\n");
	fw.write("*" + "\n");
	/*
	System.out.println("Req = "+sa.hasRequest()+"  , Confidence= "+sa.getRequestConfidence());
	System.out.println("Dlv = "+sa.hasDeliver()+"  , Confidence= "+sa.getDeliverConfidence());
System.out.println("Cmt = "+sa.hasCommit()+"  , Confidence= "+sa.getCommitConfidence());
System.out.println("Prop = "+sa.hasPropose()+"  , Confidence= "+sa.getProposeConfidence());
System.out.println("Meet = "+sa.hasMeet()+"  , Confidence= "+sa.getMeetConfidence());
System.out.println("Ddata = "+sa.hasDdata()+"  , Confidence= "+sa.getDdataConfidence());

System.out.println("*");
	*/
	
}
	fw.flush();
	fw.close();
//System.out.println("fuck");
fr.close();
}
/*
SpeechAct sa = new SpeechAct();
sa.loadMessage(msg);

System.out.println("Req = "+sa.hasRequest()+"  , Confidence= "+sa.getRequestConfidence());
System.out.println("Dlv = "+sa.hasDeliver()+"  , Confidence= "+sa.getDeliverConfidence());
System.out.println("Cmt = "+sa.hasCommit()+"  , Confidence= "+sa.getCommitConfidence());
System.out.println("Prop = "+sa.hasPropose()+"  , Confidence= "+sa.getProposeConfidence());
System.out.println("Meet = "+sa.hasMeet()+"  , Confidence= "+sa.getMeetConfidence());
System.out.println("Ddata = "+sa.hasDdata()+"  , Confidence= "+sa.getDdataConfidence());

System.out.println("*");

//another message - don't need to create another SpeechAct object. Just load the new message
String msg2 = "hi Lula, you can trust me: I will vote for your party in the next elections. I will also make sure that all congressmen approves your next bill. Best regards. Severino"; 

sa.loadMessage(msg2);

System.out.println("Req = "+sa.hasRequest()+"  , Confidence= "+sa.getRequestConfidence());
System.out.println("Dlv = "+sa.hasDeliver()+"  , Confidence= "+sa.getDeliverConfidence());
System.out.println("Cmt = "+sa.hasCommit()+"  , Confidence= "+sa.getCommitConfidence());
System.out.println("Prop = "+sa.hasPropose()+"  , Confidence= "+sa.getProposeConfidence());
System.out.println("Meet = "+sa.hasMeet()+"  , Confidence= "+sa.getMeetConfidence());
System.out.println("Ddata = "+sa.hasDdata()+"  , Confidence= "+sa.getDdataConfidence());
*/
}
}
