function converHTMLFileToPDF() {
  var doc = new jspdf.jsPDF("p", "pt", "a4");

  var source = window.document.getElementsByClassName("wrapper")[1];
  var width = doc.internal.pageSize.getWidth();
  //var source = window.document.getElementsByTagName("body")[0];

  // Convert HTML to PDF in JavaScript
  doc.html(source, {
    callback: function (doc) {
      doc.output("dataurlnewwindow");
    },
    width: width,
  });
}
