var CONFIG;
var news, tagging_article;

String.prototype.trunc = String.prototype.trunc ||
      function(n){
          return (this.length > n) ? this.substr(0, n-1) + '&hellip;' : this;
      };


function get_stats(distribution) {
  return {
    type: 'GET',
    url: url_request(CONFIG.API_STATS_URL, distribution),
    contentType: 'application/json'
  };
}

function createChart(chart, type, data, options) {
  $('#' + chart).remove(); 
  $('#chart-container').append('<canvas id="' + chart + '"></canvas>');

  var el = document.getElementById(chart);
  var ctx =  el.getContext('2d');

  return new Chart(ctx, {type, data, options});
}

function like_stats() {
  $.ajax(get_stats('like'))
  .done(function(data) {
    $('#stats_section').slideDown();
    $('#stats_title').text("like statistics")
    $('#chart-container').empty();

    var doughnut = createChart(
      'like-doughnut',
      'doughnut',
      {
        labels: ['Like', 'Dislike'],
        datasets: [{
          data: [data.num_likes, data.num_dislikes],
          backgroundColor: ['#87FC70', '#FF2A68']
        }]
      }
    );

    var barchart = createChart(
      'like-bar',
      'bar',
      {
        labels: ['Like', 'Dislike', 'Read', 'Ignored'],
        datasets: [{
          label: '# of Articles',
          data: [data.num_likes, data.num_dislikes, data.num_read, data.num_ignored],
          backgroundColor: ['#1AD6FD', '#fd0b58']
        }]
      },
      {
        responsive: true,
        scales: {
            yAxes: [{
                ticks: {
                  beginAtZero: true
                }
          }]
        },
        legend: {
          display: false
        }
      }
    );

  });
}

function tag_stats() {
  $.ajax(get_stats('tag'))
    .done(function(data) {
      $('#stats_section').slideDown();

      $('#stats_title').text("tag statistics")
      $('#chart-container').empty();

      var histogram = createChart(
        'tag-histogram', 
        'bar',
        {
          labels: data.map(a => a._id),
          datasets: [{
            label: '# of Entries',
            data: data.map(a => a.count),
            backgroundColor: '#fd0b58'
          }]
        },
        {
          responsive: true,
          scales: {
              yAxes: [{
                  ticks: {
                    beginAtZero: true
                  }
            }]
          } 
        }
      );
      
      var barchart = createChart(
        'tag-bar',
        'bar',
        {
          labels: data.map(a => a._id),
          datasets: [{
            label: '# of Likes',
            data: data.map(a => a.num_likes),
            backgroundColor: '#87FC70' // -webkit-linear-gradient(-45deg, #fd0b58 0px, #a32b68 100%);
          },
          {
            label: '# of Disikes',
            data: data.map(a => a.num_dislikes),
            backgroundColor: '#FF2A68' // -webkit-linear-gradient(-45deg, #fd0b58 0px, #a32b68 100%);            
          },
          {
            label: '# of Read',
            data: data.map(a => a.num_read),
            backgroundColor: '#1AD6FD' // -webkit-linear-gradient(-45deg, #fd0b58 0px, #a32b68 100%);            
          },
          {
            label: '# of Ignored',
            data: data.map(a => a.num_ignored),
            backgroundColor: '#e5e5e5'
          }]
        },
        {}
      );
  });
}

function hide_stats() {
  $('#stats_section').slideUp();
}

function refresh() {
  $("#page").addClass('refreshing');
  $("#refresh").show();
  $.ajax({
    type: 'PATCH',
    url: url_request(CONFIG.API_FEED_URL),
    contentType: 'application/json'
  })
  .done(function(data) {
    $("#page").removeClass('refreshing');
    $("#refresh").hide();
    loadNews();
  })
}

function getArticleIndex(id) {
  for(var i=0; i<news.length;i++)
    if(news[i]['_id'] == id) return i;
}

function stop_tagging() {
  $("#tag_section").slideUp();
}

function tagging() {
  var url = url_request(CONFIG.API_TAG_URL);

  $.get(url, function(data) {
    if($.isEmptyObject(data)) {
      alert("All articles have been tagged!");
      stop_tagging();
      return;
    }

    tagging_article = data;
    $("#tag_section").slideDown();
    $("#tag_article").html(getArticleHTMLElement(tagging_article, true));
    $("#predicted_tag").text(" " + tagging_article['predicted_tag'] + "?");
  });

}


function getArticleHTMLElement(article, tagging=false) {
  var imgUrl = (article.img == "")?"/img/news-thumbnail.jpg":article.img;
  var imgFilter = (article.img == "")?" style='filter:hue-rotate(" + Math.floor(Math.random() * 360) + "deg); '":"";
  var liked = (article.like)?" liked ": "";
  var disliked = (article.dislike)?" disliked ": "";

  var predicted_tag = (article.predicted_tag)?(article.predicted_tag + " - "): "";
  

  return  "<li data-index='" + article._id + "' class='" + liked + disliked + "'  >" +
            "<div class='card card-news'>" + 
              "<div class='list-header'>" +  
                "<img class='list-img'" + imgFilter + " src='"+ imgUrl + "'></img>" +
                "<div class='list-category'>"+article.source +"</div>" + 
                "<a class='list-title' href='javascript:void(0)'" + ( (!tagging)? "onclick='readArticle(\""+ article._id + "\")' ": "") + " >" + predicted_tag + article.title + "</a>" +
                "<div class='list-author'>"+ article.author +"</div>" + 
                "<div class='list-datetime'> <i class='fas fa-clock'></i> " + article.datetime + "</div>" + 
              "</div>" +
              "<div class='list-content'>"+ article.description.trunc(300) + "</div>" + 
              "<div class='list-footer'>" +
                ((!tagging)?"<i  onclick='like(\""+ article._id + "\")' class='far fa-thumbs-up likebtn'></i> | <i  onclick='dislike(\""+ article._id + "\")'  class='far fa-thumbs-down dislikebtn'></i>":"") + 
              "</div>" + 
            "</div>" +
          "</li>";
}

function url_request(api_url, params="") {
  params = (params!='')?'/'+params:'';
  return window.location.origin + api_url + params;
}


function ignored() {
  get_feed('ignored');
}

function liked() {
  get_feed('liked');
}

function disliked() {
  get_feed('disliked');
}

function read_articles() {
  get_feed('read');
}


function get_feed(descriptor){
  var url = url_request(CONFIG.API_FEED_URL, descriptor);
  $("#feed_section h1").text(descriptor + " articles");
  loadArticlesFromUrl(url);
}

function learning() {
  get_feed('learn')
  $("#feed_section h1").text("learning mode");
}

function like(index) {
  var li = $(event.srcElement).closest('li');
  var n = JSON.parse(JSON.stringify(news[getArticleIndex(index)]));
  var method = (!news[getArticleIndex(index)].like)?'POST':'DELETE';
  n.like = !n.like;
  n.dislike = false;

  $.ajax({
    type: method,
    url: url_request(CONFIG.API_LIKE_URL),
    data: JSON.stringify(n),
    contentType:"application/json; charset=utf-8"
  })
  .done(function(data) {
    if(method == 'POST') {
      li.removeClass('disliked');
      li.addClass('liked');
    }
    else {
      li.removeClass('liked');
    }
    news[getArticleIndex(index)] = n;
  })
  .fail(function(data) {
    alert("Error");
  });
}

function dislike(index) {
  var li = $(event.srcElement).closest('li');
  var n = JSON.parse(JSON.stringify(news[getArticleIndex(index)]));
  var method = (!news[getArticleIndex(index)].dislike)?'POST':'DELETE';
  n.dislike = !n.dislike;
  n.like = false;

  $.ajax({
    type: method,
    url: url_request(CONFIG.API_DISLIKE_URL),
    data: JSON.stringify(n),
    contentType:"application/json; charset=utf-8"
  })
  .done(function(data) {
    if(method == 'POST') {
      li.removeClass('liked');
      li.addClass('disliked');
    }
    else {
      li.removeClass('disliked');
    }
    news[getArticleIndex(index)] = n;
  })
  .fail(function(data) {
    alert("Error");
  });
}

function readArticle(index) {
  $.ajax({
    type: 'POST',
    url: url_request(CONFIG.API_READ_URL),
    data: JSON.stringify(news[getArticleIndex(index)]),
    contentType:"application/json; charset=utf-8"
  });

  var win = window.open(news[getArticleIndex(index)].link, '_blank');
  win.blur();
  window.focus();
  return false;
}

function createFeed() {
  var list = $("#feed");
  list.empty();
  for(var i=0; i<news.length;i++)
    list.append(getArticleHTMLElement(news[i]));
}

function loadArticlesFromUrl(url) {
  $.get(url, function(data) {
    news = data;
    createFeed()
  });
}

function loadNews() {
  $("#feed_section h1").text("your newsfeed");
  $("#feed").empty();
  loadArticlesFromUrl(CONFIG.API_FEED_URL);
}

function loadConfig() {
  $.getJSON('/config/config.json', function(data) {
    CONFIG = data;
    loadNews();
  });
}

function man()
{
  $("#searchbar").val('');
  var text = "";// = "<ul>";
  for(var i =0; i < CONFIG.cmds.length; i++)
  {
    text += " - " + CONFIG.cmds[i].name + ": " + CONFIG.cmds[i].description+"\n";
  }
  swal({
    title: "Lista comandi",
    text: text,
    cancel: "Got it"
  });
}

$(document).ready(function() {
  loadConfig();
  $("#searchbar").focus();

  $("#search").on('submit', function (e) {
    e.preventDefault();
    var query = $("#searchbar").val();
    var cmd = query.split(" ")[0];
    if(exec_cmd(cmd)) $("#searchbar").val(''); // if the command exists clean the command line
  });

  $("#tag_section form").on('submit', function(e){
    e.preventDefault();
    tagging_article['tag'] = $("#tag_section select").val();
    $.ajax({
      type: 'PUT',
      url: url_request(CONFIG.API_TAG_URL),
      data: JSON.stringify(tagging_article),
      contentType:"application/json; charset=utf-8"
    })
    .done(function(data) {
      tagging();
    })
    .fail(function(data) {
      alert("Error");
    });
  });
});

$(document).on('keypress', function (e) {
  if(!$("#searchbar").is(":focus") &&
     ![13,38,40,37,39].includes(e.keyCode)) {
    $("#searchbar").val('');
    $("#searchbar").focus();
  }
});

var cmd_dict = {
  'learn': learning,
  'man': man,
  'disliked': disliked,
  'liked': liked,
  'read': read_articles,
  'tag': tagging,
  'refresh': refresh,
  'feed': loadNews,
  'ignored': ignored,
  'tag-stats': tag_stats,
  'like-stats':like_stats
}

function exec_cmd(cmd) {
  if(cmd_dict[cmd] === undefined)
    return false;

  cmd_dict[cmd]();
  return true;
}