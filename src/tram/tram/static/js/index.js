
/*
<div id="{{ report.id }}" class="card shadow-sm">
    <div class="card-text">
        {{ report.name }}
        <a class="btn btn-primary btn-sm" onclick="deleteReport({{ report.id }})"><i class="fas fa-trash"></i></a>
        <div>
        Confirmed Sentences: {{ report.confirmed_sentences }}<br>
        Pending Sentences: {{ report.pending_sentences   }}<br>
        </div>
    </div>
    <div class="d-flex justify-content-between align-items-center">
        <div class="btn-group">
            <a href="/analyze/{{ report.id }}" class="btn btn-sm btn-outline-secondary">Analyze</a>
            <a href="#" class="btn btn-sm btn-outline-secondary">Export</a>
            <a href="#" class="btn btn-sm btn-outline-secondary">Download</a>
        </div>
    </div>
    <div class="card-text small">By {{ report.byline }}</div>
</div>
*/


function renderReportCard(report) {
    var confirmed = `Accepted Sentences: ${report.confirmed_sentences}`;
    var pending = `Pending Sentences: ${report.pending_sentences}`;
    var $confirmedPendingDiv = $(`<div>${confirmed}<br>${pending}<br></div>`);

    var $report = $(`<div id="report-${report.id}" class="card shadow-sm"></div>`);
    $report.append(report.name);
    $report.append(`<a class="btn btn-primary btn-sm" onclick="deleteReport(${report.id})"><i class="fas fa-trash"></i></a>`);

}