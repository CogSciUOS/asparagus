# Friday, September 6th, 2019 (Lableapp, Preprocessing)

Michael, Sophia, Malin, Ulf, Axel, Maren (Meeting in German)

## Preprocessing
* Images auf blauen Hintergrund lassen, oder Kanten Smoothen um Artefakte zu vermeiden
* Processed Images und original Bilder und Zuordnung sollen gespeichert werden (Möglichkeit unterschiedliche Varianten auszuprobieren) - Metadaten müssen gespeichert werden!
* Pipeline Preprocessing: nicht alles in einer großen Funktion, sondern in kleinst mögliche Einheiten modularisieren
* Möglichkeit einzelne Schritte ein- oder auszuschalten

* Problem Michael: bei ca. 10% der Spargelbilder fehlt eins der drei Bilder, deswegen wird Preprocessing abgebrochen und diese Bilder werden nicht zur Weiterverarbeitung verwendet
* Programme im Master sollen lauffähig sein, aber müssen nicht fertig sein. Frühere Versionen verloren gehen/nicht nutzbar sind, wenn sie nicht im Master sind
* grid benutzen, um Fehler zu reduzieren

### Preprocessing Schritte:
* Hintergrund wird entfernt, Spargel in die Mitte geschoben, bevor das Bild in drei Teile geschnitten wird
* Images heißen z.B. 1A, 1B, 1C
* jeder Spargel, hat eine eigene Nummer


## Lableapp
* Ordnerauswählbarkeit muss noch in der App integriert werden
* Bilder die nicht klassifizierbar sind in eine eigene Klasse zuordnen (Bsp: Spargel liegt quer über mehreren Schalen) "undefined"
* Bennenung der Features bleibt bei striktem Ja/Nein Antwortmöglichkeiten, da klare Definitionen für mögliche Abstufungen (Rather yes/no) schwer definierbar sind
* auch neuronales Netz lernt dann binäre Features, nicht Label 1A Anna etc. --> wir behalten mehr Variabilität
* Es wird kein extra Kopfbild des Spargels präsentiert, da Auflösung gleich bleibt
* zum Labeln von zu Hause müssen wir auf Rechner in Wachsbleiche zugreifen: linux: ffh.vs zugriff /
    - sshfs [rzlogin]@gate.ikw.uos.de [Pfad to Spargel] load folder 

## To do
* Lableapp fertig stellen
* Preprocessing: die zwei Versionen fusionieren
* Dokumentation der Tools im Preprocessing + Beispiele, wie Tool benutzt wird
* Metadaten als Tabelle:
*    Informationen: Ursprüngliches Bild, Eingetragene Lables, Klassifier Lables
* Funktion für Metadaten schreiben:
  1. Ich möchte von einem Bild auf Metadaten zugreifen
  2. Ich habe ein Bild und möchte Metadaten schreiben
* rausfinden warum das Preprocessing abgebrochen ist
* Preprocessing parallelisieren (grid)
* Lableapp soll nächste Woche fertig stellen
