function converHTMLFileToPDF() {
  var doc = new jspdf.jsPDF("portrait", "mm", "a4");

  var source = window.document.getElementsByClassName("resume")[0];
  source.style.width = "21mm";
  source.style.marginLeft = "0px";
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

function getPDF() {
  var doc = new jspdf.jsPDF({
    orientation: "p",
    unit: "mm",
    format: "a4",
    putOnlyUsedFonts: true,
  });

  // Convert HTML to PDF in JavaScript
  doc.html(document.getElementsByClassName("author__content")[0], {
    callback: function (doc) {
      doc.output("dataurlnewwindow");
    },
  });
}
