# Initial general squestions

* Sind alle Klassen immer gleich definiert? Oder gibt es einstellbare Parameter (z.B. in einem schlechten Erntejahr 
ist die Minimaldicke für die beste Klasse geringer oder so? Oder sogar innerhalb einer Saison variabel?)
* Welche Klassen gibt es und wie sind sie definiert? (Da sollten wir wirklich eine Liste haben, wo tabellarisch aufgeführt ist
wie die Klasse heißt und wie sie definiert ist)
* Generell Informationen über die Klassen, gibt es eine Hierarchie (z.B. die Klassen A, B und C sind nur feinere Variationen
von Klasse D)
* welche der Klassen sollten in jedem Fall beibehalten werden? Auf welche könnte verzichtet werden? 
* Was ist die Verarbeitungsgeschwindigkeit der Maschine (in Spargelstangen pro Sekunde)? Können wir die manipulieren? 
(ggf brauchen wir länger als 1/5 Sekunde pro Stange)
* Was ist die a-priori-Wahrscheinlichkeitsdistribution über die Klassen? Also was sind die Prozentangaben (ungefähr) 
mit denen sich die Stangen auf die 14 Klassen verteilen?

* Wie läuft die Software zur Zeit auf der Maschine?
* Was können wir als Marker im "background" verwenden? 
* Erhebt die Maschine noch zusätzliche Informationen zu jeder Stange (z.B. Breite, Krümmungsgrad ...), die theoretisch abgespeichert werden könnten? 
* Was ist Silvan's bisherige Intuition, wie "gut" die Maschine ist? Heißt wie viel Prozent richtig sortiert werden.



## Beschreibung des Datenerhebungsprozesses

Der Spargel läuft durch die Maschine und wird sortiert. Menschliche Arbeiter korrigieren die Sortierung, sodass jetzt in 
jeder Kiste nur Spargelstangen liegen, die zu der entsprechenden Klasse gehören. 

Die Kisten, die so gefüllt wurden, werden aus der Maschine entfernt (sodass sie in den folgenden Schritten nicht "verunreinigt"
werden).
Jede dieser Kisten wird nun einzeln durch die Maschine gefüttert, z.B. alle Spargel der Klasse 1. 
Die dabei entstandenen Bilder müssen in einem Ordner abgelegt werden, der „Klasse 1“ oder so heißt, oder jede Datei muss 
so benannt werden dass ersichtlich ist zu welcher Klasse sie gehört, auf jeden Fall müssen wir für jedes Bild wissen, was 
für Spargelstangen darauf sind. Dass auf diese Weise mehrere Stangen pro Bild erscheinen ist in Ordnung, solange sie zur
gleichen Klasse gehören.
Wenn eine Kiste durchgelaufen ist, sind also die in ihr enthaltenen Spargelstangen wieder auf die entsprechenden Ausgänge
verteilt. Wenn die Maschine perfekt sortieren würde, lägen sie alle in einer Kiste, aber da die Maschine Fehler macht, liegen 
manche Stangen jetzt in einer anderen Kiste. Idealerweise notieren wir, wie diese neue Verteilung jetzt aussieht (also z.B. von
1000 Stangen der Klasse 1 werden 850 korrekt in Klasse 1 sortiert, 100 landen in Klasse 2, 25 in Klasse 7 und 25 in Klasse 8.)
Diese Information wäre wichtig, um zu wissen, welche Klassen die Maschine mit welchen anderen Klassen verwechselt, weil wir so 
die Fehler unseres Systems mit denen der Maschine vergleichen können. Wenn das zu viel Aufwand ist, wäre es trotzdem sehr gut, 
wenn wir wenigstens wüssten, wie viele Stangen die Maschine richtig gemacht hat (in diesem Fall 850/1000 für Klasse 1). 
Diese Information ist extrem wichtig, weil sie uns Aufschluss über die Qualität der Sortierung der Maschine gibt. Wenn wir diese 
Information nicht haben, unser System aber testen und 87% richtig machen, wissen wir nicht, ob das jetzt gut oder schlecht war, 
weil wir keine Baseline Performance haben. 

Idealerweise haben wir also nachher 14 Ordner (einer für jede Klasse) mit jeweils einem Haufen Bilder drin, auf denen
nur Spargel dieser Klasse zu sehen sind. Zusätzlich füllen wir eine Tabelle:

| Klasse | Anzahl | Sortiert zu Klasse 1 | Sortiert zu Klasse 2 | Sortiert zu Klasse 3 | Sortiert zu Klasse 4 | Sortiert zu Klasse 5 |
| ------ | ------ | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| 1      | 10500  | 10000                | 250                  | 3                    | 2                    | 245                  |
| 2      | 7000   | 0                    | 6900                 | 0                    | 50                   | 50                   |
| 3      | 12300  | 20                   | 80                   | 1200                 | 0                    | 0                    |

usw. 
