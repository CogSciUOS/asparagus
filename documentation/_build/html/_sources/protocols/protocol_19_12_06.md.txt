# Friday, December 6th, 2019 (ANNs, PCA, image transfer)
Anwesend: Axel, Ulf, Malin, Maren, Josefine  

## Neural Networks

-	Clustering auf auto encoder laufen lassen?
-	Zuerst PCA anwenden macht Sinn
-	Generative adversarial network: vielleicht nützlich wenn man neue Spargelbilder generieren will, ansonsten nicht unbedingt; interessant, wenn man gute Bilder hat; man füttert den Code in das erste Netzwerk, generiert ein Bild und das zweite Netzwerk gibt dann “ja”/”nein” aus; besser als zweiter/dritter/vierter Schritt gedacht
-	Beim Semi-Supervised Learning würde ab und an das ‘Feature’ (also Label) mit reingeworfen werden.
-	Der Auto Encoder Approach wäre unsupervised, es entstehen freie Cluster; man gewinnt eine komprimierte Darstellung der Daten; nicht das “Learning” ist interessant bei Machine Learning sondern was für Feature Darstellungen/Gruppierungen entstehen
-	Voll Supervised Learning passt mein Model so an das es zuverlässig meine Cluster ausgibt
-	Semi-Supervised Learning versucht Code zu extrahieren (z.B. Loss Function: wenn ich Label habe will ich wissen, wie diese aussehen und die Daten ohne Label will ich möglichst gut rekonstruieren)

## Gelabelte Daten

-	Wir haben jetzt ca. 5000-6000 gelabelte Bilder.

## Grid und Preprocessor

-	Grid Manager
-	Preprocessor verbessert
-	Verbesserung von Feature Extractor für Features wie “Krumm”, jetzt gibt es mehr Werte die das aussagen aus denen man den eigentlichen Code rausbekommen könnte
-	Feature für violett funktioniert auch noch nicht immer (vielleicht threshold höher setzen?)
-	Submit Script vorgestellt

## Ziele nächste Woche

-	Transfer der Ordner von Sommer- zu Wintersemester
-	Erst PCA fertig zum laufen kriegen,
-	dann neural encoder,
-	weiter labeln,
-	und für nach Weihnachten nochmal ein Kappa Agreement machen und die Accuracy und Einheitlichkeit prüfen.
