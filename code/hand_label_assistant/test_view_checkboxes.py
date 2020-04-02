from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

model = QStandardItemModel()
item = QStandardItem("Item")
item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
item.setData(QVariant(Qt.Checked), Qt.CheckStateRole)
model.appendRow(item)



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    view = QListView()
    view.setModel(model)
    view.show()
    sys.exit(app.exec_())
