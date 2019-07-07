# Friday, 05. July 2019 (Data sorting, Labeling App, Feature Extraction Methods, Holiday Organisation)

Michael, Richard, Malin, Maren, Sophia, Josefine, Axel und Ulf  
(Meeting in German)

## Zusammenfassung letzte Woche
-	Schritte, was wir machen können
-	Ordner sortieren für Datenbasis
-	Preprocessing (anfangen)
-	Michael hat die Preprocessing Methode/ Label App verbessert, sieht aus als sollte es (bald) nutzbar sein
-	müsste noch machen: aus Datei rauslesen, sodass man die Spargelklassen benennen kann
-	überlegen, wieviele handgelabelte Daten wir haben wollen
-	wenn Bilder sortiert sind, können wir schauen, wieviele gelabelte Daten wir schon haben (und ob es schon geht, weiterzuarbeiten mit den bereits gelabelten Daten)
-	Zugriff auf Bilder (Uni-Server): Sortieren der Ordner ist etwas langsam
-	Es ist noch eine Festplatte in Rheine angeschlossen
  
### Datensortierung
Bilder sortieren
-	Jedes Bild hat F00, F01 oder F02 am Namensanfang und suggertiert Zusammengehörigkeit dieser Bilder zu einem Spargel
-	Dann gibt es eine Zahl, die einzigartig ist für jeden Spargel (die dann bei allen drei Bildern steht)
-	Spargel, der leicht quer auf den Schalen liegt, wird zur Zeit weggeschnitten; den Rahmen aber gleich zu behalten macht weniger Arbeit auf die Anzahl an Spargeln, bei denen das der Fall ist, gesehen; können die Bilder erstmal in eigenen Ordner schieben und später überlegen, was wir damit machen

Labelling App
-	Eingetragene Zeichen sind gerade kleine Zahlen, vielleicht kann mand as noch ändern zu leeren Flächen, NaN, oder -1
-	Bevor wir die App starten müssen wir die Feature Extraction (Dicke etc.) noch anpassen, um Sortieren zu erleichtern
-	Idee für Report am Ende: Vergleich von Labelling von Mensch und Maschine
-	Viele der Feature Extraction Methoden sind schon fertig: Rost, Blume und Violett sind noch nicht fertig

Labelling + Feature Extraction
-	Wie sinnvoll ist es die App jetzt schon zu benutzen und bauen die Features später ein? – Zwei verschiedene Ansätze, hat Vor- und Nachteile
-	Klassifikation per Auge, gemischt mit Klassifikation von Feature Extraction Methode

Wieviele gelabelte Daten brauchen wir?
-	Je mehr Parameter, desto mehr Daten werden gebraucht, es gibt keine pauschale Antwort auf Frage, wieviele Trainingsdaten wir brauchen (e.g. ImageNet 1000 Bilder pro Klasse)
-	Durch Data Augmentation bekomment wir auch noch mehr
-	Vielleicht erstmal ‘fauler’ Ansatz und 100 gelabelte Bilder von jeder Klasse nehmen, augmentieren sodass es 500 pro Klasse sind und einfach schonmal testen, was passiert
  
## Organisatorisches
Github Sortierung: Feature Extraction
-	Manual Feature Extractor: Github Ordner zum Sammeln der Feature Extraction Methoden, dass z.B. die Labeling App extern darauf zugreifen kann und nicht Code manuell eingefüg wird
-	App in Github pushen
-	Neues Repository beinhaltet alle Dateien, aber mit neuer Sortierung der Ordner

Arbeitseinteilung (im Sommer)
-	Block Sessions in den Ferien abhalten, in denen wir am Stück zusammen am Projekt arbeiten
-	Modus für Semesterferien zu überlegen macht Sinn
-	Wöchentliche Treffen mit Ulf und Axel machen nur Sinn, wenn es auch etwas zu besprechen gibt
-	Schedule auf dem Google Kalender soll Auskunft über Anwesenheit von Teilnehmern geben
-	Erste/zweite Septemberwoche als Blockwoche eingetragen

Allgemeine Punkte von Ulf
-	Lehrevaluationsbogen noch ausfüllen, damit er ausgewertet werden kann
-	Es werden noch Tutoren gesucht für die Computer Vision Lecture

Asparagus Classification Report
-	Wurde bereits begonnen zu schreiben, ist aber offen für alle und soll auch gerne von allen genutzt werden
-	Bericht zum Notieren, was wir bereits gemacht haben
