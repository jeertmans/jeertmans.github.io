import { jsPDF } from "jsPDF";

var doc = new jsPDF();

var source = window.document.getElementsByClassName("body")[0];
gt;
doc.fromHTML(source);
doc.output("dataurlnewwindow");
