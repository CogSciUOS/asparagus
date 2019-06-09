24.05.19  
8.30h - 11.30h  
  
Attendants: Thomas, Maren, Sophia, Josefine, Katharina, Richard, Manuel, Tobias, Ulf, Axel, Thomas Hermeler  

#### Disclaimer: most will be written in German

# Meeting with Mr. Hermeler

## Introduction
  
Presentation by Thomas
-	background of cognitive science
-	backgroung machine learning & computervision → image classification
-	we don’t know the labels of the machine
  
## Q&A with Mr. Hermeler
  
#### Q: How does the machine classify
-	when the machine saves a picture, it is already processed (features were already made and it was sorted)
-	Entscheidungsparameter, die gesetzt werden (können sich jeden Tag ändern), hängen vom Kunden ab und können angepasst werden
-	wenn man von Klassifizierung spricht sind diese fest gesetzt, die Grenzen verschwimmen allerdings
-	wo messe ich den Spargel (oben, unten, mitte) für Durchmesser?
-	aus den drei Bildern wird ein Mittelwert für die Klassifizierung gemacht; wenn da die Parameter an den grenzen liegen, welche Klassifizierung gilt dann?
- man kann auch sagen, ich nehme nur den dicksten oder dünnsten - wo genau will ich messen?
-	guter Spargel ist meist oben etwas dünner und unten etwas dicker (solche Dinge kann man miteinbeziehen)
     
#### Q: Gibt es mechanische Probleme (dass die Maschine nicht richtig angeschlossen ist)? Wir haben die Demoversion qvd demo images gestartet. Wenn man OK bei der Fehlermeldung drückt schließt sich das ganze Programm. Woran liegt das?
-	vllcht fehlt noch eine Datei
-	heißt theoretisch, dass die Demo nicht gestartet wurde
  
#### Q: Kann man die Feature, die die Maschine erstellt, rauslesen? Wir wollen sichergehen, ob z.B. der Durchmesser der gleiche für Maschine wie für unsere Berechnung ist/ dass wir diesen auf die gleiche Art bestimmen.
-	früher war es so: 1A Spargel wird durchgeschickt und anhand dessen die Parameter für diesen festgemacht usw.
-	- Randparameter (wo ich messe), kann man momentan auch eingeben, liegt aber alles daran, wie man es programmiert
-	bestimmte sachen treten nur in den ersten saisontagen auf, welche effektiver per Hand zu lösen sind
  
#### Q: Müssen die neuen Maschinen noch manuell eingestellt werden?
- ja
-	Kameraauswertung macht die Software Firma; die Firma macht normalerweise Rohlinge von Verpackungen (PET Flaschen müssen an Verschraubung geprüft werden)
  
#### Q: Bleibt das Programm gleich bzw. kommt alles noch von der gleichen Maschine?
-	bin jedes Jahr dabei das Programm zu verbessern
-	kann man schon alles nicht mehr auswerten mit den Netzwerkstrukturen
-	ich will etwas erkennen, dass nur 4 Pixel groß ist (zB eine Farbe), das wird schwierig; Farbton nur erkennbar sobald dieser auch eindeutig mind. einen Pixel hat
-	bei den neueren Maschinen gibt es eine zweite Kamera nur allein für den Kopf
-	die Auflösung ist dann bei 1 zu 16 (also viel besser)
-	ein Grund ist, dass vor Jahren Kameras teurer waren, aber heute eine günstige Lösung geworden sind
-	aber eine Kamera für 20 000€ einsetzen, das will der Kunde dann auch nicht (zu teuer)
-	Tendenz: Kapazität oder Rechenleistung ist kein begrenzender faktor mehr

#### Q: Was macht deine Firma genau, die Maschinen bauen oder nur warten?
-	beides

#### Q: Sie haben Kontakt zu der Firma in Aachen?
-	ja

#### Q: Könnte man direkt mit der Kamera “sprechen”?
-	das ist outgesourced, da gibt es von mir keine Informationen
-	‘Ich will die Daten anders von der Kamera bekommen’, das kann ich nicht machen

#### Q: Zugriff auf die Kamera oder Dokumentation wäre interessant gewesen.
-	das werden die nicht rausgeben

#### Q: Gibt es eine Doku der API (Hardware ist nicht unser Fachgebiet)? Speziell geht es um die Schnittstelle zwischen Programm und ‘öffne die-und-die Klappe’.
-	im PC ist die Schnittstelle zu API data karte
-	eine Doku für (elektrische) Signalübertragung ist da
-	die Doku könnt ihr haben, aber damit könnt ihr vllcht nicht viel anfangen
-	ich bin gerade dabei ein anderes Protokoll zu schreiben, bin da aber wieder an den Hersteller der Software gebunden
-	die Firma Intravis hat die Schnittstelle geschrieben und diese Dokumentation werden wir wieder nicht bekommen

#### Q: Wenn wir ihnen unseren Code geben, könnte diese Firma uns eine Schnittstelle bauen?
-	das wäre vllcht eine Möglichkeit

#### Q: Zurück zur neuen Kamera: wieviel bringt eine hohe Auflösung für die Klassifizierung?
-	doppelt so viel - wobei wir hatten eben nur über durchmesser geredet
-	mit der Farbe wird es schwerer: wenn die Farbe jetzt zu 90% richtig ist, ist sie das mit der neuen Kamera nachher zu 98% (deutliche Steigerung)
-	Aufblüher sind da ein anderes Problem, da müsste man schon mit einer zweiten Kamera arbeiten
-	2D Bild und 3D klassifizieren ist eine Problematik, die ich mit zwei Kameras gelöst habe
-	mit 2 Kameras zu arbeiten ist aber ein komplett neuer Bereich, der gerade am kommen ist (Firma Klaas macht das schon seit Jahren)

#### Q: Wieviele Hersteller für Spargelsortiermaschinen gibt es überhaupt?
-	so 4 bis 5?


#### Q: Wie sind die Probleme bei anderen Kunden? Wie hoch ist die Fehlerwahrscheinlichkeit? Solange der Unterschied nicht riesig ist, sind Kundenspezifische Klassifizierungsunterschiede nicht das Problem. Problematisch sind die, die so falsch klassifiziert sind, das es auffällt.
-	da gibt es auch große Unterschiede (zB entscheidend ist die Kopfform), das werde ich aber mit der einen Kamera nicht gelöst bekommen
-	oder die Beleuchtung ist nicht optimal
-	bei neueren Maschinen muss nicht ganz so viel nachjustiert werden, aber immer noch ab und an mal
-	Spargel mit Macke nur auf einer Seite erwischt man unter Umständen nicht

#### Q: Beim Warten, was genau stellen Sie da ein? Manche Sachen kann man natürlich nicht umstellen, aber was lässt sich erfolgreich an Parmetern verstellen?
-	ich stelle die Parameter jedes mal ein (der Kunde kann das nicht)
-	es muss immer (wieder) angepasst werden
-	ob ich die Auflagen des kunden auf grund der parameter einstellen und verbessern kann, müsste ich mal schauen
-	‘Es muss doch einfach einzustellen sein.’ - in der Regel, wenns einmal eingestellt ist, wird da kein Käufer mehr was ändern
-	wenn es nicht auftritt, kann ich das auch nicht anpassen
-	‘Farbe passt nicht.’ - Hat sich vllcht die Beleuchtung oder die Farbe geändert?
-	ich kann verschiedene Farbwerte bekommen, abhängig vom Licht draußen (Sonnenschein etc) → eine Lösung war z.B. hier LED zu benutzen, aber es ändert sich immer noch (wenn auch weniger)
-	die Hardware ist schon die eine Sache, sonst kann man das mit der Software nicht mehr ändern

#### Q: Ist eine perfekte Klassifizierung möglich, bzw. ähnlich wie wenn ein Mensch sich die Fotos angucken würde? Kriegt man da eine 95%ige Genauigkeit?
-	nein

#### Q: Was ist das Limit der Maschine?
-	90%
-	Schalen müssten täglich gereinigt werden (an Ecken sammelt sich Dreck)
-	mit Pech ist das die gleiche Farbe wie bei dunklem, violetten Spargel

#### Q: Als Mensch sehen ich ja der Dreck an der Maschine ist nicht auf dem Spargel.
-	auch dann kann man die Farbe nicht immer eindeutig erkennen

#### Q: Wenn ein Mensch das erkennen kann und wir haben genug Bilder (theoretisch kommt man da schon hin) wäre für uns interessant: ist es auch möglich mit den Bildern die wir haben? Was ist das Maximum das wir erwarten können?
-	ca 90%, auch mit so vielen Bildern
-	ich habe Zuhause eine Testmaschine
-	wir können den Spargel da durchjagen, dafür könnt ihr gerne vorbeikommen

#### Q: Gibt es eine allgemein universelle Klassifikation? Oder ist das immer kundenabhängig?
-	von der Landwirtschaftskammer gibt es eine Klassifizierung

#### Q: Solange es in sich konsistent ist, sind dann die meisten Kunden zufrieden? Wo ist liegt das Hauptaugenmerk?
-	der Durchmesser muss passen, alles andere machen die Leute dann viel von Hand
-	bei neuen Maschinen bin ich bei 98% den korrekt zu treffen
-	Sauberkeit ist ein Problem
-	im mittleren Bereich der Anzeige auf der Maschine gibt es eine Parameterliste (auf life Bild links angezeigt), da bekomme ich bisschen die Statistik her
-	da kann man einigermaßen erkennen, warum hat die Maschine was wo reingeworfen oder warum nicht
-	da steht nicht wieviel Prozent, sondern nur ‘leicht geöffnet’
-	extremes Beispiel ist rostiger Spargel (Kunde sagt, er hat damit kein Problem, um dan festzustellen, dass 90% Rostspargel ist..)
-	die Definition ‘leicht geöffnet’, ‘leicht gerostet’, hängt sehr vom Kunden ab (alles sehr relativ)
-	Klassifizierungsreihenfolge (Krümmung, violett etc) ist bei jedem Kunden anders prioritisiert
-	wie die Maschine das macht: alle Parameter die passen kommen in eine Exceltabelle auf die ich Zugriff habe
-	ich setze z.B. erstmal alle Klassifizierungen weiß bevor violett kommt
-	anderes Beispiel wäre Krümmung zuzulassen bei leicht violettem Spargel, weil man da mehr Krümmung zulässt als bei weißem Spargel
-	am einfachsten ist selber Spargel drauflegen und das Klassifizieren macht die Maschine schon selber

#### Q: Also entstehen alle Parameter beim ersten Durchlauf?
-	meine erste Sortierung lass ich durchlaufen und Parameter entstehen

#### Q: Also grundsätzlich habe ich alle Parameter/Klassen von Anfang an abgespeichert?
-	da denken sie an ein anderes System
-	man geht eig genauso hin: ‘erste Sortierung’ durchlaufen lassen, und danach muss man entstandene Parameter anpassen

#### Q: Wie weit weichen bundesweite Klassifizierungen ab?
-	extrem
-	andere könnten Spargel hier nicht verkaufen
-	‘wann ist ein Spargel violett?’, im süden muss er extrem weiß sein, da sind mehr Klassifizierungen notwendig
-	Köln ist ca die Nord-Süd Grenze
-	es gibt zwar diese ganzen Vorgaben, aber das einzige, was so wirklich funtioniert, ist der Durchmesser. Aber auch der spaltet sich so langsam auf.
-	dem Kunden klarzumachen wie er zu arbeiten hat, das ist die Schwierigkeit für mich
-	Toleranzgrenze: wie kriege ich die wieder raus, wie kann ich die wieder abziehen von meinen Parametern?
-	der größte Fehler liegt oft beim Kunden
-	obwohl das Programm alles vorgibt kann die Maschine die richtigen Parameter nicht finden. Der kunde steht aber nur 2 mal im Jahr wirklich davor (und guckt nach den Parametern)
-	die Varianz bei Spargel ist extrem groß

#### Q:Es macht also Sinn deutlich mehr Klassen als 14 aufzustellen und die später zusammenzulegen.
- zb tiptop 1a Spargel und ganz okay 1a Spargel. Die dann zusammenzuschmeißen ist abhängig von Saison/Kunde/Tag
-	habe ich son in meiner Maschine, der Kunde versteht das System (und die dazugehörige Problematik) aber meist nicht
-	in der Software ist es so installiert, dass erst weißer, dann violetter Spargel kommt

#### Q: Also erst Durchschnittswerte berechnen, dann in die Tabelle damit und dann kommt die Sortierung. Wenn ich sehe, dass der Durchschnitt 20 war, jetzt aber 22 ist,  dann muss ich eins höher gehen?
-	ja

#### Q: Wenn wir theoretisch eine (halb)perfekte Klassifizierung haben (aber auch da gibt es noch Nachsortierung): gibt es da Kommunikation zwischen Software und Maschine, also, zählt die Maschine wie voll die Boxen werden können?
-	die Maschine hat 16 Fächer, 15 sind schaltbar
- bei mehr ist auch mehr Sortierung möglich

#### Q: Es geht mehr darum, ob das Fach schon voll ist. Wird überladen?
-	ja
-	man kann schon einstellen, wenn man weiß ich hab mehr 1a Spargel, dann nimmt man dafür 2 Fächer
-	vor Jahren dachte man mal, die Maschine muss möglichst flexibel sein; dabei waren letztlich die Nachsortierer nicht ganz so flexibel

#### Q: Wie stabil ist die Klassifizierung? Wenn man die gleiche Charge nochmal durchschickt wie gleich wird die Klassifizierung sein?
-	so 95%
-	auch da gehts los mit Durchmesser, Farbe etc
- es werden leicht andere Werte berechnet
-	Reproduzierbarkeit hat nunmal das Problem (der Datenvarianz), aber 90% kann man schon erwarten

#### Q: Die Maschine nutzt alle 3 Bilder pro Spargel?
-	ja
-	Auflösung, Drehen des Spargels ect.. die Pixel der Kamera stellen z.B. eine Grenze dar (Kunde: ‘Wieso liegt der Spargel einmal bei 16 und dann bei 12-16?’)
-	ich kann Bilder abspeichern und mir nacher angucken, dann Parameter verändern und schauen ob mein Spargel dahin kommt, wo er hinsoll
-	mein größtes Problem ist: der Kunde lässt gerne mal Spargel zu, den die Maschine innerhalb der Grenzen nicht zulassen würde (laut Kunde darf bei 10 guten auch mal ein schlechter rein - und dann kommen plötzlich 100 Schlechte rein, weil die Maschine das nicht versteht)

#### Q: Haben Sie eigene Probebilder und wären die verfügbar?
-	ich kann ihnen nochmal bessere Bilder bereitstellen
-	vor allem für die Köpfe
-	ich habe ca 1000-10 000 Bilder verfügbar in die Richtung
-	für meine Auswertung ziehe ich Bilder von Kunden via Teamviewer runter und sortiere/kontrolliere die dann im Büro; das Anpassen geht so viel besser/schneller
-	auch ein Problem: jeder Spargel ist anders

#### Q: Die Bilder die sie haben, sind also von unterschiedlichen Bauernhöfen?
-	ja
-	einziger Unterschied für mich ist: ich muss die Software jedes mal etwas anders anpassen (Farbdefinition kann bei jedem Standort unterschiedlich ausfallen)

#### Q: Wie stark ist der saisonale Unterschied? Macht eine saisonale (also jährliche) Prüfung am Anfang Sinn?
-	eigentlich bleibt alles sehr stabil
-	wir nutzen LED Beleuchtung, hier ist aber auch altersbedingt, dass die schlechter wird/anders beleuchtet
-	ich muss die Kamera öfter mal normen
-	wenn der Kunde ja sagt, dann kann man das schon machen, es muss nur alles öfter mal überprüft werden
-	das Anpassen der Maschine dauert in der Regel den ganzen Tag, da das zeitlich verteilt werden muss (Maschine laufen lassen, Bilder angucken, wieder laufen lassen ...)
-	und der Kunde muss bereit sein die Maschine für eine Prüfung abzuschalten

#### Q: Wir haben seit neuestem eine ‘kein Bild vorhanden’ Fehlermeldung, wo dann die Maschine abbricht.
-	Steckerverbindung an Kamera und Kameraleitung zu PC lösen und wieder reinmachen
-	ist ein Korosionsproblem
-	ansonsten ist evtll die Kamera über Ethernet mit dem PC verbunden und im PC gibt es eine Steckkarte, die auch Korosion auslösen kann. Die müsste man dann lösen und wieder reinstecken
-	pc problem, da feuchte räume pc korosion fördert
-	am besten immer nur den Hauptschalter ausmachen und alle Stecker der Maschine drinlassen (um Feuchtigkeit zu vermeiden)

#### Q: Können wir die Klassifizierung von ihren Parametern haben?
-	ja

#### Q: Gibt es Konkurrenz bei der Software?
-	ja, es gibt nicht nur einen Anbieter

#### Q: Was machen andere Hersteller anders?
-	zB Prüfung von hohlem Spargel, da ist die Trefferquote allerdings nur bei 50% bisher
-	was muss ich investieren
-	doppelspargel krieg ich da zB nicht
-	muss viel know-how reinstecken, über Schattenbildung kommt man schon mit 85%er Genauigkeit hin was Aufblüher bretrifft; frage aber immernoch, was man mit falsch deklarierten macht
-	wenn ich nicht bei 90-95% Genauigkeit bin muss man eh nachsortieren
-	Aufblüher sind das beste b´Beispiel für diese aÁrt von Problemen

#### Q: Was ist ihr persönlicher Hintergrund und wie ist so grob die Spargelsortiermaschinengeschichte?
-	das Spargelsortieren per Maschine ist seit ca 2000 da (die Maschine ist von 2003)
-	mein Vater hatte selber 2-3 ha Spargel
-	Spargel ist eines der schwersten Produkte zur Sortierung; es gibt kaum ein Produkt, das so viele Klassen hat wie Spargel

#### Q: Wie stellen sie sich eine Kooperation vor?
-	ich bin bereit das hier zu unterstützen (auch finanziell)
-	wo die Grenzen sind, speziell bei der Maschine, ist halt die Kameraauflösung - die Parameter reichen dafür nämlich nicht
-	in der Hinsicht würde ich euch unterstützen
- was brauchen Sie an Unterstützung? Ich bin da recht frei

#### Q: Was ist gerade so der Standard was die Kamera betrifft?
-	aus Kostengründen werden auch immernoch gerne einfachere Kameras genommen
-	Hochauflösung wäre zwar wünschenswert, viele arbeiten aber noch mit den schlechten (es ist nicht mal eine 50-50 Aufteilung von alten vs. neuen Maschinen auf dem Markt)
-	ich stelle gerne Bilder zur Verfügung mit entsprechend guter Qualität
-	die alte Maschine nutzen ist halt schwer, weil die ihre Grenzen hat (zB jeden Tag Schalen putzen kostet auch Geld; genauso sollte auch die Kamera gereinigt werden)
-	man kann in der Maschine eigentlich alle Schritte nachvollziehen: wann ein Spargel nicht erkannt wird, wann da Unsicherheiten sind; es gibt Kunden da läuft die Maschine mit hälfte roter Striche

#### Q: Nicht nur Schalen, sondern auch Kamera?
-	wie mache ich die Kamera sauber? Muss man auch schauen, wie man die putzt, dass das Bild nicht unscharf wird
-	gibt alle Möglichkeiten der Prüfung
-	ich bin schon am Überlegen bei neuen Maschinen Putzdüsen einzubauen, damit das richtig gemacht wird

#### Q: Nochmal zur Kooperation: was erhoffen sie sich?
-	das was sie da machen: dass ich das Programm, das ich jetzt habe, wegtun kann und stattdessen Ihres nehme
-	sie machen ja ein neues Programm mit komplett neuer Klassifizierung usw
-	die Ausgangs-/Maschinensteuerung sollte kein Problem sein (Kanbussteuerung?); geht über Ethernet, wann welche Klappe aufmacht (Ergebnis wird in Schieberegister geschoben, dieses Register wird bei jedem Takt um eins verändert – ist alles noch Softwarebereich)
-	Schnittstelle zwischen PC und Maschine ist nix wildes

#### Q: Bei Fragen können wir nochmal auf Sie zukommen? Z.B. bei der Hochkategorisierung ist eine Zusammenfassung meist leichter: was wie leichter/schwerer ist (aus praxissicht) da können wir noch auf Sie zukommen?
-	an sich, ja
-	zusammenwerfen mache ich ja zB schon

#### Q: Kann man jemandem der auflegt noch eine Aufgabe mitgeben?
-	dauert deutlich länger
-	zu hohes Tempo

#### Q: Die Demoversion gestartet zu kriegen wäre z.B. schon ein erster Ansatz für eine Kooperation für uns. Ist einen Screenshot mit der Fehlermeldung zu schicken möglich?
-	den Ordner htms(?) runterladen via Teamviewer, daraus dann starten mit den Bilern die du schon hast; sollte auf jeden Rechner draufspielbar sein
-	Klassifizierungssoftware liegt dort
-	ich kann auch andere Maschinen, mehr/andere Bilder (von anderen Maschinen) bereitstellen
-	1 Bild sind ca. 4 Megabyte

#### Q: Optimal wäre eine Klassifizierung von den Bildern wo erkennbar ist welches Bild von welchem Hof stammt. Wäre das möglich?
-	wäre vllcht möglich
-	Sie sind eingeladen sich den Betrieb mal anzuschauen
-	habe schon drei Bachelorabsolventen gehabt (war aber mit Schwerpunt Mechanik)
-	die Software ist ausgesourced, ich habe keinen (wirklichen) Programmierer in der Firma dabei

## Abschlussdiskussion
  
Möglichkeit der RAM Verbesserung?
-	vllcht können wir neuen RAM ein/dazubauen in den PC
  
Wäre eine Möglichkeit bei Silvan's Maschine eine zweite Kamera dazuzubauen? Mit wenigen Kosten z.B. (go pro + raspberry) oder ändern wir nichts mehr an der Hardware?
-	ist wahrscheinlich zu schwierig
-	unser Setup ist eigentlich richtig blöd, aber wir haben ncht mehr viel Zeit und müssen die Bilder nehmen die wir jetzt haben
-	vllcht zweites Projekt eröffnen um das Setup zu verbessern
-	Kamerakalibrierung sollten wir nochmal ansprechen
-	vllcht eine Checkliste für die Maschine bei jedem Saisonstart? (Kamera Kalibrierung, Putzen etc)
   
Kamerakalibrierung mit Sticker?
-	Sticker auf Schalen und eine als Kalibrierungsschale nutzen; jeden Durchlauf nachkalibrieren (Stickergröße und Farbe wären fest) → Autosticker nutzen (reflektieren nicht zu doll und wären einigermaßen wasserfest); vllcht bis nächste Woche Freitag überlegen, was wir drucken wollen und wo wir das her haben wollen
  
Blumenerkennung schwierig wegen evenutell zu schlecher Kamera
-	Spargelkopf ist etwas verschwommen gewesen

Zugriff auf kamera wäre gut
  
  
Telefonat mit Mr Hermeler
-	er kann bei anderem Kunden fragen, ob wir die Kopfbilder der zweiten Kamera sammeln dürfen für den Rest der Saison
-	würden dann die Software von Thomas installieren, dass wir die Bilder direkt hochladen können
-	es wäre Wert das zu probieren
-	Terminvereinbarung
-	eine Frage wäre wie anders die Programme laufen (wieviele Bilder könnten wir bekommen?)

Exkursion zur anderen Spargelmaschine/hof (falls wir dort hindürfen) am Mittwoch (29.05.19)
- zweites Telefonat hat ergeben, dass es keine Internetverbindugn auf besagtem zweiten Hof gibt und wir somit leider nicht das Programm von Thomas nutzen können
- Anzahl der Bilder wäre zu gering; müsste jedes mal von Hand gesammelt werden
- Exkursion steht vllcht noch an, aber ist somit auch nicht dringend
- eine neue Terminvereinbarung wird überlegt
