/*-------------------------------------
* Functionality for uploading a report
*-------------------------------------*/

//When the Upload button is clicked, call the function to upload a report
$(document).on('change', '#id-upload',function (event){
    if(event.currentTarget) {
        let filename = event.currentTarget.files[0].name;
        if(filename){
            uploadReport();
        }
    }
});


//POST the report to the endpoint
function uploadReport() {
    console.log('in uploadreport')
    let file = document.getElementById('id-upload').files[0];
    let csrf_token = document.getElementById('id-upload-form').querySelector('[name="csrfmiddlewaretoken"]').value;
    let fd = new FormData();
    console.log(csrf_token)
    fd.append('csrfmiddlewaretoken', csrf_token);
    fd.append('file', file);
    $.ajax({
         type: 'POST',
         url: '/upload/',
         data: fd,
         processData: false,
         contentType: false,
         success: function(response) {
            console.log(response)
         },
    }).done(function (){
        location.reload();
    })
}

function deleteJob(report_id) {
    $.ajax({
        type: 'DELETE',
        url: 'api/jobs/' + report_id + '/',
        headers: {
            "X-CSRFToken": CSRF_TOKEN
        }
    }).done(function(){
        location.reload()
    })
}

function deleteReport(report_id) {
    $.ajax({
        type: 'DELETE',
        url: 'api/reports/' + report_id + '/',
        headers: {
            "X-CSRFToken": CSRF_TOKEN
        }
    }).done(function(){
        location.reload()
    })
}